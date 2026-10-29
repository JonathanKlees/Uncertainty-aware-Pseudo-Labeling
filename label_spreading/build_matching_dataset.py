"""Builds the `matched_dataset` parquet for Cityscapes/KITTI/VOC: each Grounding
DINO prediction is greedily IoU-matched (1:1, by descending score) to the soft-GT
objects of its image and stored with its soft label, embedding and class probs.
Expects the soft-GT json in COCO's schema (see build_soft_gt.py).

COCO val and COCO train were matched with build_matching_dataset.ipynb /
build_matching_dataset_coco_train.ipynb instead, because GDino's COCO labels are
sparse COCO category ids (not a dense 0..N-1 index) and COCO val additionally
records the match to the standard COCO GT.

`gdino_label` is read off the soft-GT's own categories list (by the predicted
class index), so it is spelled exactly the way compute_map's `name_to_id`
lookup expects.

The optional `clip_embedding` column (used only by analysis/ablation.ipynb's
CLIP-vs-GDino-embedding comparison) is carried through when present: run
inference/clip_embeddings.py on --predictions_path first to add it.
"""
import os
import json
import argparse

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def iou_matrix(boxes1, boxes2):
    """(N,4) x (M,4) [x1,y1,x2,y2] -> (N,M) IoU matrix."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)))
    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    areas1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    areas2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = areas1[:, None] + areas2[None, :] - inter
    return inter / np.maximum(union, 1e-6)


def greedy_match(pred_scores, iou_mat, iou_thr):
    """Greedy 1-1 matching sorted by descending score.

    Returns match_iou (N,) float and match_idx (N,) int (-1 if unmatched)."""
    N, M = iou_mat.shape
    matched_gt = np.zeros(M, dtype=bool)
    match_iou = np.zeros(N, dtype=np.float32)
    match_idx = np.full(N, -1, dtype=int)
    for i in np.argsort(pred_scores)[::-1]:
        row = iou_mat[i].copy()
        row[matched_gt] = 0.0
        best = int(row.argmax())
        if row[best] >= iou_thr:
            match_iou[i] = row[best]
            match_idx[i] = best
            matched_gt[best] = True
    return match_iou, match_idx


def build_matched_dataset(predictions_path, soft_gt_path, output_path,
                           iou_thr=0.5, score_thr=0.0):
    df_pred = pd.read_parquet(predictions_path)
    if score_thr > 0:
        df_pred = df_pred[df_pred["score"] >= score_thr].reset_index(drop=True)
    print(f"Predictions : {len(df_pred):,} detections across {df_pred['image_id'].nunique():,} images")

    # optional column, present only if clip_embeddings.py was run on predictions_path
    has_clip = "clip_embedding" in df_pred.columns

    with open(soft_gt_path) as f:
        soft_gt = json.load(f)
    categories = soft_gt["categories"]                      # includes trailing "cantsolve"
    n_classes = len(categories) - 1                          # real classes, excluding cantsolve
    idx_to_name = {i: cat["name"] for i, cat in enumerate(categories)}
    fname_to_imgid = {img["file_name"]: img["id"] for img in soft_gt["images"]}

    soft_by_imgid = {}
    for fname, objs in soft_gt["objects"].items():
        img_id = fname_to_imgid.get(fname)
        if img_id is not None:
            soft_by_imgid[img_id] = objs
    print(f"Soft GT     : {len(soft_by_imgid):,} images with annotations  |  {n_classes} classes + cantsolve")

    zero_soft = np.zeros(n_classes + 1, dtype=np.float32)
    rows = []

    for img_id, grp in tqdm(df_pred.groupby("image_id"), desc="Matching images"):
        pred_boxes = np.array(grp["box"].tolist(), dtype=np.float32)     # (N,4) xyxy
        pred_scores = grp["score"].values.astype(np.float32)
        pred_labels = grp["label"].values.astype(np.int32)               # dense 0..N-1 class index
        pred_embs = grp["last_hidden_state"].tolist()                     # (256,) decoder hidden state
        pred_probs = np.vstack(grp["probs"].to_numpy()).astype(np.float32)  # (N, n_classes), already dense
        pred_clip = grp["clip_embedding"].tolist() if has_clip else None   # (512,), optional
        N = len(pred_boxes)

        soft_objs = soft_by_imgid.get(img_id, [])
        match_iou = np.zeros(N, dtype=np.float32)
        match_idx = np.full(N, -1, dtype=int)
        if soft_objs:
            soft_boxes = np.array([o["bbox"] for o in soft_objs], dtype=np.float32)
            mat = iou_matrix(pred_boxes, soft_boxes)
            match_iou, match_idx = greedy_match(pred_scores, mat, iou_thr)

        for i in range(N):
            s_idx = int(match_idx[i])
            if s_idx >= 0:
                sobj = soft_objs[s_idx]
                bbox_xyxy = sobj["bbox"]
                bbox = [bbox_xyxy[0], bbox_xyxy[1],
                        bbox_xyxy[2] - bbox_xyxy[0], bbox_xyxy[3] - bbox_xyxy[1]]
                soft_label = np.array(sobj["soft_label"], dtype=np.float32)
                soft_label_mod = np.array(sobj["postprocessed_soft_label"], dtype=np.float32)
                label = int(np.argmax(soft_label))
            else:
                bbox = None
                soft_label = zero_soft.copy()
                soft_label_mod = zero_soft.copy()
                label = n_classes   # cantsolve / background

            row = {
                "image_id": img_id,
                "gdino_pred_id": i + 1,
                "gdino_bbox": pred_boxes[i].tolist(),
                "gdino_score": float(pred_scores[i]),
                "gdino_label": idx_to_name[int(pred_labels[i])],
                "gdino_embedding": pred_embs[i],
                "gdino_probs": pred_probs[i],
                "gdino_match": int(s_idx >= 0),
                "gdino_bbox_iou": float(match_iou[i]),
                "bbox": bbox,
                "soft_label": soft_label,
                "soft_label_mod": soft_label_mod,
                "label": label,
            }
            if has_clip:
                row["clip_embedding"] = pred_clip[i]
            rows.append(row)

    df = pd.DataFrame(rows)
    print(f"\nDataset shape : {df.shape}")
    print(f"Soft GT match : {df['gdino_match'].sum():,} / {len(df):,}  ({df['gdino_match'].mean():.1%})")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    print(f"Saved -> {output_path}  ({os.path.getsize(output_path) / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--soft_gt_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--iou_thr", type=float, default=0.5)
    parser.add_argument("--score_thr", type=float, default=0.0)
    args = parser.parse_args()
    build_matched_dataset(args.predictions_path, args.soft_gt_path, args.output_path,
                           iou_thr=args.iou_thr, score_thr=args.score_thr)
