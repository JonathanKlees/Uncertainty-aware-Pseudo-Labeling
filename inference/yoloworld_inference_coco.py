import argparse
import os
import sys
import time
import json
from tqdm import tqdm
import cv2
import torch
import numpy as np
import pandas as pd

from ultralytics import YOLOWorld

from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils.nms import non_max_suppression
from ultralytics.utils.ops import scale_boxes
from ultralytics.nn.modules.head import WorldDetect

from ultralytics.engine.results import Results
###
# NOTE: Patches Ultralytics NMS function in YOLO Post-processing to return class probabilities.
# Patch was written for Ultralytics 8.4.22 (Python 3.10.20); imports may need adjusting for newer versions.
###

# Captures the per-anchor embedding fed into the contrastive class head (cv3 output, before
# cv4). WorldDetect has no transformer decoder either; this is the vision-side embedding
# compared against text embeddings to produce class scores for each anchor, i.e. the closest
# analogue of a per-query decoder hidden state. Keyed by scale index (WorldDetect.cv3[0..nl-1]).
_captured_cv3 = {}

def _make_cv3_hook(scale_idx):
    def _hook(module, input, output):
        _captured_cv3[scale_idx] = output.detach()  # [batch, embed_dim, H_i, W_i]
    return _hook

class CustomYOLOPredictor(DetectionPredictor):

    def postprocess(self, preds, img, orig_imgs):

        preds_nms, indices = non_max_suppression(
            preds,
            conf_thres=self.args.conf,
            iou_thres=self.args.iou,
            max_det=self.args.max_det,
            return_idxs=True
        )

        # Reconstruct per-anchor hidden states in the exact order WorldDetect.forward
        # concatenates anchors (scale 0..nl-1, each flattened in raster order), so indexing
        # with the NMS-kept indices lines up with boxes/scores/probs.
        hs_scales = [_captured_cv3[i] for i in sorted(_captured_cv3)]
        bs_, embed_dim = hs_scales[0].shape[0], hs_scales[0].shape[1]
        hidden_states = torch.cat([h.view(bs_, embed_dim, -1) for h in hs_scales], dim=2)
        hidden_states = hidden_states.permute(0, 2, 1)  # [bs, num_anchors, embed_dim]

        results = []

        for i, (pred, idx) in enumerate(zip(preds_nms, indices)):
            orig_img = orig_imgs[i]
            path = self.batch[0][i]

            if pred is None or len(pred) == 0:
                empty_boxes = torch.zeros((0, 6), device=img.device)

                r = Results(
                    orig_img=orig_img,
                    path=path,
                    names=self.model.names,
                    boxes=empty_boxes
                )

                r.class_probs = np.zeros((0, len(self.model.names)))
                r.last_hidden_state = np.zeros((0, embed_dim))

            else:
                # Scale boxes back to original image size
                # pred[:, :4] = x1, y1, x2, y2 in model input size
                pred[:, :4] = scale_boxes(
                    img.shape[2:],  # model input H,W
                    pred[:, :4],    # boxes in model input
                    orig_img.shape[:2]  # original image H,W
                ).round()

                r = Results(
                    orig_img=orig_img,
                    path=path,
                    names=self.model.names,
                    boxes=pred
                )

                raw = preds[i][0]              # [C, N]
                raw = raw[:, idx.long()]      # [C, n]
                raw = raw.T                  # [n, C]

                r.class_probs = raw[:, 4:].cpu().numpy()
                r.last_hidden_state = hidden_states[i][idx.long()].cpu().numpy()

            results.append(r)

        return results

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "YOLOv8 eval on COCO", add_help=True)

    # COCO image directory
    parser.add_argument("--image_dir", type=str,
                        required=True, help="coco image dir")
    
    parser.add_argument("--model_name", type=str, default="yolov8l-worldv2", help="name of the model")

    parser.add_argument("--save_path", type=str, default="predictions/yolov8l-worldv2/coco/predictions.parquet",
                        help="path for the output parquet file (directory is created automatically)")

    args = parser.parse_args()

    id_map = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18, 17: 19, 18: 20, 19: 21, 20: 22, 21: 23, 22: 24, 23: 25, 24: 27, 25: 28, 26: 31, 27: 32, 28: 33, 29: 34, 30: 35, 31: 36, 32: 37, 33: 38, 34: 39, 35: 40, 36: 41, 37: 42, 38: 43, 39: 44, 40: 46,
                        41: 47, 42: 48, 43: 49, 44: 50, 45: 51, 46: 52, 47: 53, 48: 54, 49: 55, 50: 56, 51: 57, 52: 58, 53: 59, 54: 60, 55: 61, 56: 62, 57: 63, 58: 64, 59: 65, 60: 67, 61: 70, 62: 72, 63: 73, 64: 74, 65: 75, 66: 76, 67: 77, 68: 78, 69: 79, 70: 80, 71: 81, 72: 82, 73: 84, 74: 85, 75: 86, 76: 87, 77: 88, 78: 89, 79: 90}

    # load COCO pre-trained YOLOv8 model
    model = YOLOWorld(f"checkpoints/{args.model_name}.pt")

    # Apply Patch

    model.predictor = CustomYOLOPredictor(overrides=model.overrides)
    model.predictor.setup_model(model=model.model)

    # Register hooks to capture the per-anchor embeddings (cv3 output) used by the
    # contrastive class head; see _make_cv3_hook / _captured_cv3 above.
    head = None
    for m in model.model.modules():
        if isinstance(m, WorldDetect):
            head = m
            break
    assert head is not None, "Could not locate WorldDetect head in model"
    _hook_handles = [
        head.cv3[i].register_forward_hook(_make_cv3_hook(i))
        for i in range(len(head.cv3))
    ]

    # Inference on COCO. batch=1 keeps `preds[i][0]` in CustomYOLOPredictor.postprocess
    # (and our hidden-state indexing) aligned with the current image, since AutoBackend
    # returns [y_tensor, preds_dict] rather than a per-image list for batch_size > 1.
    results = model.predict(
        source = args.image_dir,
        conf = 0.001,
        iou = 0.7,
        max_det = 300,
        batch = 1,
        stream=False, # we store ourselves in COCO format
    )

    for h in _hook_handles:
        h.remove()

    # accumulate per-detection rows for the parquet file
    all_rows = []
    for r in tqdm(results):
        image_id = int(os.path.basename(r.path).split(".")[0])

        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes  = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        labels = r.boxes.cls.flatten().cpu().numpy()
        probs  = r.class_probs        # new attribute from the patch
        hidden = r.last_hidden_state  # new attribute from the patch

        for j in range(scores.shape[0]):
            row = {
                "image_id"         : image_id,
                "score"            : float(scores[j]),
                "label"            : id_map[int(labels[j])],  # map to COCO category IDs
                "box"              : boxes[j].tolist(),        # [x1,y1,x2,y2]
                "probs"            : probs[j],                 # per-class probabilities
                "last_hidden_state": hidden[j],
            }
            all_rows.append(row)

    # write all detections to a single parquet file
    print(f"\nSaving {len(all_rows)} detections to {args.save_path} ...")
    df = pd.DataFrame(all_rows)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    df.to_parquet(args.save_path, engine="pyarrow", compression="snappy", index=False)
    print("Done.")