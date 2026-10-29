"""Filters an already-computed predictions parquet down to a specific image-ID
subset, e.g. SoftTeacher's semi-supervised split annotations.

Used to derive `coco_train_fold1_1pct/predictions.parquet` for OWL-ViT and
YOLO-World from their full COCO train predictions (`coco_train/predictions.parquet`):
`analysis/ablation.ipynb`'s OVOD-ensemble section calibrates/compares only on
the SoftTeacher fold=1/1%-budget labeled images (1182 of the 118k), the only
images with ground truth to calibrate against.

Usage:
    python filter_predictions_by_image_subset.py \\
        --predictions_path predictions/owlvit/coco_train/predictions.parquet \\
        --labels_json ${HOME}/SoftTeacher/data/coco/annotations/semi_supervised/instances_train2017.1@1.json \\
        --output_path predictions/owlvit/coco_train_fold1_1pct/predictions.parquet
"""
import argparse
import json
import os

import pandas as pd


def filter_by_image_subset(predictions_path, labels_json, output_path):
    with open(labels_json) as f:
        labels = json.load(f)
    image_ids = set(img["id"] for img in labels["images"])
    print(f"{labels_json}: {len(image_ids):,} images")

    df = pd.read_parquet(predictions_path)
    print(f"{predictions_path}: {len(df):,} detections across {df['image_id'].nunique():,} images")

    df = df[df["image_id"].isin(image_ids)].reset_index(drop=True)
    print(f"Filtered: {len(df):,} detections across {df['image_id'].nunique():,} images")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", required=True, help="full-scope predictions parquet to filter")
    parser.add_argument("--labels_json", required=True,
                         help="COCO-format annotation json whose 'images' list defines the subset to keep")
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    filter_by_image_subset(args.predictions_path, args.labels_json, args.output_path)
