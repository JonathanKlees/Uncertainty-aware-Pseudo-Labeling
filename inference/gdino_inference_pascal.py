import argparse
import os
import sys
import time
import json
from PIL import Image

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler

from groundingdino.models import build_model
import groundingdino.datasets.transforms as T
from groundingdino.util import box_ops, get_tokenlizer
from groundingdino.util.misc import clean_state_dict, collate_fn
from groundingdino.util.slconfig import SLConfig

import torchvision

from groundingdino.util.vl_utils import build_captions_and_token_span, create_positive_map_from_span
from groundingdino.datasets.cocogrounding_eval import CocoGroundingEvaluator


def load_model(model_config_path: str, model_checkpoint_path: str, device: str = "cuda"):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    model.eval()
    return model

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(self, img_folder, ann_file, transforms):
        super().__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        img, target = super().__getitem__(idx)  # target: list

        w, h = img.size
        boxes = [obj["bbox"] for obj in target]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]  # xywh -> xyxy
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)
        # filter invalid boxes/masks/keypoints
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]

        target_new = {}
        image_id = self.ids[idx]
        target_new["image_id"] = image_id
        target_new["boxes"] = boxes
        target_new["orig_size"] = torch.as_tensor([int(h), int(w)])

        if self._transforms is not None:
            img, target = self._transforms(img, target_new)

        return img, target

class PostProcessCocoGrounding(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""

    def __init__(self, num_select=300, cat_list=None, tokenlizer=None) -> None:
        super().__init__()
        self.num_select = num_select

        assert cat_list is not None
        captions, cat2tokenspan = build_captions_and_token_span(cat_list, True)
        tokenspanlist = [cat2tokenspan[cat] for cat in cat_list]
        positive_map = create_positive_map_from_span(
            tokenlizer(captions), tokenspanlist)  # 80, 256. normed

        id_map = {i : i for i in range(20)} # Identity mapping for PascalVOC 20 classes.
        # build a mapping from label_id to pos_map
        new_pos_map = torch.zeros((20, 256))
        for k, v in id_map.items():
            new_pos_map[v] = positive_map[k]
        self.positive_map = new_pos_map

    @torch.no_grad()
    def forward(self, outputs, target_sizes, not_to_xyxy=False, last_hidden_state=None):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
            last_hidden_state: optional [batch, num_queries, hidden_dim] decoder hidden states captured
                                via forward hook; if provided, the top-k selected states are returned
                                in each result dict under the key "last_hidden_state".
        """
        num_select = self.num_select
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        # pos map to logit
        prob_to_token = out_logits.sigmoid()  # bs, 100, 256
        pos_maps = self.positive_map.to(prob_to_token.device)
        # (bs, 100, 256) @ (20, 256).T -> (bs, 100, 20)
        prob_to_label = prob_to_token @ pos_maps.T

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = prob_to_label
        topk_values, topk_indexes = torch.topk(
            prob.view(out_logits.shape[0], -1), num_select, dim=1)
        scores = topk_values
        topk_boxes = topk_indexes // prob.shape[2]
        labels = topk_indexes % prob.shape[2]

        if not_to_xyxy:
            boxes = out_bbox
        else:
            boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)

        boxes = torch.gather(
            boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

        # Gather top-k class probability distributions
        probs_of_selected_queries = torch.gather(
            prob,
            1,
            topk_boxes.unsqueeze(-1).expand(-1, -1, prob.shape[2])
        )

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        # Gather top-k decoder hidden states using the same topk_boxes indices
        if last_hidden_state is not None:
            selected_hs = torch.gather(
                last_hidden_state,
                1,
                topk_boxes.unsqueeze(-1).expand(-1, -1, last_hidden_state.shape[2])
            )
            results = [{'scores': s, 'labels': l, 'boxes': b, 'probs': p, 'last_hidden_state': h}
                       for s, l, b, p, h in zip(scores, labels, boxes, probs_of_selected_queries, selected_hs)]
        else:
            results = [{'scores': s, 'labels': l, 'boxes': b, "probs": p}
                       for s, l, b, p in zip(scores, labels, boxes, probs_of_selected_queries)]

        return results

def main(args):
    # config
    cfg = SLConfig.fromfile(args.config_file)

    # build model
    model = load_model(args.config_file, args.checkpoint_path)
    model = model.to(args.device)
    model = model.eval()

    # Hook to capture the last decoder hidden state (hs[-1]) without modifying the library.
    # model.transformer returns (hs, references, hs_enc, ref_enc, init_box_proposal);
    # hs has shape [num_layers, batch, num_queries, hidden_dim], so output[0][-1] is the
    # final decoder layer's output for all queries.
    _captured_hs = {}
    def _hs_hook(module, input, output):
        _captured_hs['value'] = output[0][-1].detach()  # [batch, num_queries, hidden_dim]
    _hook_handle = model.transformer.register_forward_hook(_hs_hook)

    # build coco style dataset and dataloader
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    cat_list=[
            "aeroplane","bicycle","bird","boat","bottle",
            "bus","car","cat","chair","cow",
            "diningtable","dog","horse","motorbike","person",
            "pottedplant","sheep","sofa","train","tvmonitor"
        ]

    if not os.path.exists(args.anno_path): # if no coco style annotation file exists
        print("No COCO style annotation file found. Please create one using create_coco_style_annotations.ipynb ...")


    dataset = CocoDetection(
        args.image_dir, args.anno_path, transforms=transform)
    data_loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    # build captions
    caption = " . ".join(cat_list) + ' .'
    print("Input text prompt:", caption)

    # build post processor
    tokenlizer = get_tokenlizer.get_tokenlizer(cfg.text_encoder_type)
    postprocessor = PostProcessCocoGrounding(
        cat_list=cat_list, tokenlizer=tokenlizer)


    # run inference
    all_rows = []  # accumulates one dict per detection for final parquet write
    start = time.time()
    for i, (images, targets) in enumerate(data_loader):

        # get images and captions
        images = images.tensors.to(args.device)
        bs = images.shape[0]
        input_captions = [caption] * bs

        # feed to the model; the transformer hook populates _captured_hs['value']
        outputs = model(images, captions=input_captions)

        orig_target_sizes = torch.stack(
            [t["orig_size"] for t in targets], dim=0).to(images.device)

        # pass the captured hidden state so the postprocessor can gather the
        # top-k selected query embeddings alongside boxes/scores/probs
        results = postprocessor(
            outputs, orig_target_sizes,
            last_hidden_state=_captured_hs.get('value')
        )

        # accumulate per-detection rows for the parquet file
        for target, res in zip(targets, results):
            image_id = target["image_id"]
            boxes  = res["boxes"].cpu()
            scores = res["scores"].cpu()
            labels = res["labels"].cpu()
            probs  = res["probs"].cpu()
            hidden = res.get("last_hidden_state")
            if hidden is not None:
                hidden = hidden.cpu()

            for j in range(scores.shape[0]):
                row = {
                    "image_id"         : image_id,
                    "score"            : float(scores[j]),
                    "label"            : int(labels[j]),
                    "box"              : boxes[j].tolist(),        # [x1,y1,x2,y2]
                    "probs"            : probs[j].numpy(),         # per-class probabilities
                    "last_hidden_state": hidden[j].numpy() if hidden is not None else None,
                }
                all_rows.append(row)

        if (i+1) % 30 == 0:
            used_time = time.time() - start
            eta = len(data_loader) / (i+1e-5) * used_time - used_time
            print(
                f"processed {i+1}/{len(data_loader)} images. time: {used_time:.2f}s, ETA: {eta:.2f}s")

    _hook_handle.remove()  # clean up the forward hook

    # write all detections to a single parquet file
    print(f"\nSaving {len(all_rows)} detections to {args.save_path} ...")
    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    df.to_parquet(args.save_path, engine="pyarrow", compression="snappy", index=False)
    print("Done.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "Grounding DINO eval on PascalVOC", add_help=True)
    # load model
    parser.add_argument("--config_file", "-c", type=str,
                        required=True, help="path to config file")
    parser.add_argument(
        "--checkpoint_path", "-p", type=str, required=True, help="path to checkpoint file"
    )
    parser.add_argument("--device", type=str, default="cuda",
                        help="running device (default: cuda)")

    # post processing
    parser.add_argument("--num_select", type=int, default=300,
                        help="number of topk to select")

    # dataset info
    parser.add_argument("--image_dir", type=str,
                        required=True, help="dataset image dir")
    parser.add_argument("--anno_path", type=str,
                        required=True, help="path to the COCO style annotation json file")

    parser.add_argument("--num_workers", type=int, default=4,
                        help="number of workers for dataloader")
    
    parser.add_argument("--save_path", type=str, default="predictions/gdino/pascal/predictions.parquet",
                        help="path for the output parquet file (directory is created automatically)")
    args = parser.parse_args()

    main(args)