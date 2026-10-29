"""One-off harmonization of the Cityscapes/KITTI/VOC soft-GT jsons so they all
share COCO's schema (top-level `categories`/`images`/`objects`, `objects` keyed
by the exact `file_name` string used in `images`) -- the schema
`evaluation.py`'s `_load_hard_gt_from_soft`/`_load_soft_gt_by_image` expect.

Two problems in the raw crowd-sourced soft-GT files this fixes:
  - Cityscapes' and KITTI's top-level `classes` (list of strings) instead of
    `categories` (list of {id, name} dicts), and Cityscapes has no `images`
    list at all.
  - Cityscapes/KITTI `objects` keys are bare filename stems ("000178"),
    while KITTI's own (partial) `images` list and the COCO-format annotation
    files use the full filename ("000178.png") -- a straight lookup
    mismatch.

`images` in the output are taken from a COCO-format annotation file whose
`image_id`s match the inference predictions: `soft_instances_*.json` for
Cityscapes/KITTI (the COCO-format version of the soft-label benchmark's own
boxes, from `our_labels/`) and `instances_trainval.json` for VOC. (The
`instances_*.json` files for Cityscapes/KITTI are the datasets' official
annotations, used only by the label-error-detection study.) All of these are
created by inference/create_coco_style_annotations.ipynb. `categories` come
from the original soft-GT's `classes`/`categories` (which include the trailing
`cantsolve` background class the detection-only annotation files don't), and
`objects` are merged from the per-split soft-GT source files, remapped onto the
same filenames.
"""
import json
import os


def _load(path):
    with open(path) as f:
        return json.load(f)


def _categories_from_source(soft_gt):
    if "categories" in soft_gt:
        return soft_gt["categories"]
    return [{"id": i, "name": name} for i, name in enumerate(soft_gt["classes"])]


def build_harmonized_soft_gt(dataset_name, our_instances_path, soft_gt_source_paths,
                              output_path, filename_fix=None):
    our_instances = _load(our_instances_path)
    images = [{"file_name": im["file_name"], "id": im["id"],
               "width": im["width"], "height": im["height"]}
              for im in our_instances["images"]]
    valid_filenames = {im["file_name"] for im in images}

    categories = None
    objects = {}
    for path in soft_gt_source_paths:
        soft_gt = _load(path)
        if categories is None:
            categories = _categories_from_source(soft_gt)
        for stem, objs in soft_gt["objects"].items():
            fname = filename_fix(stem) if filename_fix else stem
            if fname in objects:
                raise ValueError(f"duplicate filename '{fname}' while merging {soft_gt_source_paths}")
            objects[fname] = objs

    missing = valid_filenames - objects.keys()
    unmatched = objects.keys() - valid_filenames
    print(f"{dataset_name}: {len(images)} images, {len(objects)} object entries "
          f"({len(missing)} images with no GT objects, {len(unmatched)} GT entries with no matching image)")

    objects = {fname: objs for fname, objs in objects.items() if fname in valid_filenames}

    out = {"dataset_name": dataset_name, "categories": categories, "images": images, "objects": objects}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(out, f)
    print(f"  -> saved {output_path}")


if __name__ == "__main__":
    SOFT_GT_DIR = "../data/soft_datasets"

    build_harmonized_soft_gt(
        "Cityscapes_trainval",
        "/home/klees/data/Cityscapes/trainval/soft_instances_trainval.json",
        [f"{SOFT_GT_DIR}/soft_Cityscapes_train.json", f"{SOFT_GT_DIR}/soft_Cityscapes_val.json"],
        f"{SOFT_GT_DIR}/soft_Cityscapes_trainval_fixed.json",
        filename_fix=lambda stem: stem + ".png",
    )

    build_harmonized_soft_gt(
        "Kitti_2D_train",
        "/home/klees/data/KITTI/train/soft_instances_train.json",
        [f"{SOFT_GT_DIR}/soft_Kitti_2D_train.json"],
        f"{SOFT_GT_DIR}/soft_Kitti_2D_train_fixed.json",
        filename_fix=lambda stem: stem + ".png",
    )

    build_harmonized_soft_gt(
        "PascalVOC_2012_detection_trainval",
        "/home/klees/data/VOC/VOCdevkit/VOC2012/instances_trainval.json",
        [f"{SOFT_GT_DIR}/soft_PascalVOC_2012_detection_train.json",
         f"{SOFT_GT_DIR}/soft_PascalVOC_2012_detection_val.json"],
        f"{SOFT_GT_DIR}/soft_PascalVOC_2012_detection_trainval_fixed.json",
    )
