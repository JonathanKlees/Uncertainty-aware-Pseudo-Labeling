"""Embeds the image crop of every detection in an OVOD's predictions parquet
(as produced by e.g. gdino_inference_coco.py) with a CLIP image encoder, so
the resulting embedding space can be used as a drop-in alternative to
`gdino_embedding` (the decoder hidden state) for the label-spreading graph in
../label_spreading -- see build_matching_dataset.ipynb.

Reads predictions/<model>/<dataset>/predictions.parquet (--model selects which
detector's predictions to read, default "gdino"), crops each detection's box out
of the corresponding source image, encodes every crop with CLIP, and augments
the SAME parquet in place with a new `clip_embedding` column (row order/every
other column unchanged) -- written atomically (temp file + rename) so a crash
mid-write can't corrupt the original predictions file.
"""

import argparse
import os
import time

import clip
import numpy as np
import pandas as pd
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


def _clip_box(x1, y1, x2, y2, w, h):
    """Clamps a [x1,y1,x2,y2] box to the image bounds, guaranteeing a
    non-empty (>=1px) crop even if rounding/GDino's box regression placed it
    exactly on (or past) the image edge."""
    x1 = min(max(x1, 0.0), w - 1)
    y1 = min(max(y1, 0.0), h - 1)
    x2 = min(max(x2, x1 + 1), w)
    y2 = min(max(y2, y1 + 1), h)
    return x1, y1, x2, y2


class ImageCropDataset(Dataset):
    """One item per image: opens/decodes the image once and crops every one
    of its detection boxes, instead of one item per detection (which would
    redecode the same JPEG up to `num_select` times)."""

    def __init__(self, df, image_dir, imgid_to_fname, preprocess):
        self.df = df
        self.image_dir = image_dir
        self.imgid_to_fname = imgid_to_fname
        self.preprocess = preprocess
        self.groups = list(df.groupby("image_id", sort=False).groups.items())

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        image_id, row_idx = self.groups[i]
        row_idx = np.asarray(row_idx)
        fname = self.imgid_to_fname[int(image_id)]
        img = Image.open(os.path.join(self.image_dir, fname)).convert("RGB")
        w, h = img.size

        crops = []
        for idx in row_idx:
            x1, y1, x2, y2 = self.df.at[idx, "box"]
            x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, w, h)
            crops.append(self.preprocess(img.crop((x1, y1, x2, y2))))

        return torch.stack(crops), row_idx


def _collate_single_image(batch):
    (crops, row_idx), = batch   # batch_size=1 -- one image (all its crops) per item
    return crops, row_idx


def main(args):
    device = args.device

    print(f"Loading CLIP model {args.clip_model} ...")
    model, preprocess = clip.load(
        args.clip_model, device=device, download_root=os.environ.get("MODEL_CACHE_DIR"),
    )
    model.eval()
    embed_dim = model.visual.output_dim
    param_dtype = next(model.parameters()).dtype

    predictions_path = args.predictions_path or f"predictions/{args.model}/{args.dataset}/predictions.parquet"
    save_path = args.save_path or predictions_path   # default: augment the source file in place

    if args.score_thr > 0 and os.path.abspath(save_path) == os.path.abspath(predictions_path):
        raise ValueError(
            "--score_thr > 0 would drop the filtered-out rows from the canonical predictions "
            "file when writing back in place -- pass --save_path to write a separate (filtered) "
            "file instead, or drop --score_thr to embed (and keep) every row."
        )

    print(f"Loading {args.model} predictions from {predictions_path} ...")
    df = pd.read_parquet(predictions_path)   # full frame: every column is carried through to the output
    if args.score_thr > 0:
        df = df[df["score"] >= args.score_thr].reset_index(drop=True)
    print(f"Predictions : {len(df):,} detections across {df['image_id'].nunique():,} images")

    coco = COCO(args.anno_path)
    imgid_to_fname = {img["id"]: img["file_name"] for img in coco.dataset["images"]}

    dataset = ImageCropDataset(df, args.image_dir, imgid_to_fname, preprocess)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers,
        collate_fn=_collate_single_image,
    )

    embeddings = np.zeros((len(df), embed_dim), dtype=np.float32)

    start = time.time()
    with torch.no_grad():
        for i, (crops, row_idx) in enumerate(tqdm(loader, desc="Embedding crops")):
            for s in range(0, len(row_idx), args.batch_size):
                chunk = crops[s:s + args.batch_size].to(device=device, dtype=param_dtype)
                feats = model.encode_image(chunk).float().cpu().numpy()
                embeddings[row_idx[s:s + args.batch_size]] = feats

            if (i + 1) % 500 == 0:
                used_time = time.time() - start
                eta = len(loader) / (i + 1e-5) * used_time - used_time
                print(f"processed {i + 1}/{len(loader)} images. time: {used_time:.2f}s, ETA: {eta:.2f}s")

    df["clip_embedding"] = list(embeddings)

    print(f"\nSaving {len(df):,} rows (+clip_embedding) to {save_path} ...")
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    # atomic write: build the full file under a temp name first, then rename it over the
    # target in one step -- if this process dies mid-write, the original file (or a
    # previous temp file) is left intact instead of a truncated/corrupt parquet.
    tmp_path = save_path + ".tmp"
    df.to_parquet(tmp_path, engine="pyarrow", compression="snappy", index=False)
    os.replace(tmp_path, save_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CLIP crop embeddings for OVOD predictions", add_help=True)

    # which detector's predictions to embed (predictions/<model>/<dataset>/)
    parser.add_argument("--model", type=str, default="gdino",
                         help="detector whose predictions to read (predictions/<model>/<dataset>/predictions.parquet)")
    parser.add_argument("--dataset", type=str, default="coco",
                         help="dataset subfolder under predictions/<model>/ (e.g. coco, coco_train)")
    parser.add_argument("--predictions_path", type=str, default=None,
                         help="explicit path to the predictions parquet, overrides --model/--dataset")
    parser.add_argument("--score_thr", type=float, default=0.0,
                         help="pre-filter predictions below this score before embedding (0 = keep all)")

    # CLIP model
    parser.add_argument("--clip_model", type=str, default="ViT-B/32",
                         help="CLIP model name, see clip.available_models()")
    parser.add_argument("--device", type=str, default="cuda", help="running device (default: cuda)")
    parser.add_argument("--batch_size", type=int, default=256,
                         help="max crops per encode_image call (an image can have more detections than this)")
    parser.add_argument("--num_workers", type=int, default=4,
                         help="number of workers for image loading/cropping/preprocessing")

    # source images, for resolving image_id -> file_name and cropping boxes
    parser.add_argument("--anno_path", type=str, required=True, help="COCO-format annotation json")
    parser.add_argument("--image_dir", type=str, required=True, help="root directory of the source images")

    parser.add_argument("--save_path", type=str, default=None,
                         help="output parquet path (default: overwrite the source predictions file in place)")

    args = parser.parse_args()

    main(args)
