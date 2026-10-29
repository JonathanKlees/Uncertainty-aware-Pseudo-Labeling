#!/bin/bash
# Run from this directory (inference/) -- paths below are relative to it.
# Each block can take hours; COCO (118k images) dominates wall-clock. Run
# under tmux/nohup, not interactively.

### Grounding DINO

# grounding dino inference on coco validation data
CUDA_VISIBLE_DEVICES=0 python gdino_inference_coco.py \
    -c configs/GroundingDINO_SwinB_cfg.py \
    -p checkpoints/groundingdino_swinb_cogcoor.pth \
    --anno_path ${HOME}/data/coco/annotations/instances_val2017.json \
    --image_dir ${HOME}/data/coco/val2017 \
    --save_path predictions/gdino/coco/predictions.parquet

# grounding dino inference on coco training data (needed for pseudo-label generation)
CUDA_VISIBLE_DEVICES=0 python gdino_inference_coco.py \
    -c configs/GroundingDINO_SwinB_cfg.py \
    -p checkpoints/groundingdino_swinb_cogcoor.pth \
    --anno_path ${HOME}/data/coco/annotations/instances_train2017.json \
    --image_dir ${HOME}/data/coco/train2017 \
    --save_path predictions/gdino/coco_train/predictions.parquet

# grounding dino inference on coco unlabeled data (needed for the 100%-budget
# pseudo-label setting, generate_pls_pseudo_labels_100pct.py)
CUDA_VISIBLE_DEVICES=0 python gdino_inference_coco.py \
    -c configs/GroundingDINO_SwinB_cfg.py \
    -p checkpoints/groundingdino_swinb_cogcoor.pth \
    --anno_path ${HOME}/data/coco/annotations/instances_unlabeled2017.json \
    --image_dir ${HOME}/data/coco/unlabeled2017 \
    --save_path predictions/gdino/coco_unlabeled/predictions.parquet


# grounding dino inference on cityscapes (train+val merged into one trainval split)
CUDA_VISIBLE_DEVICES=0 python gdino_inference_cityscapes.py \
    -c configs/GroundingDINO_SwinB_cfg.py \
    -p checkpoints/groundingdino_swinb_cogcoor.pth \
    --anno_path ${HOME}/data/Cityscapes/trainval/soft_instances_trainval.json \
    --image_dir ${HOME}/data/Cityscapes/trainval/images

# grounding dino inference on pascalVOC trainval data
CUDA_VISIBLE_DEVICES=0 python gdino_inference_pascal.py \
    -c configs/GroundingDINO_SwinB_cfg.py \
    -p checkpoints/groundingdino_swinb_cogcoor.pth \
    --anno_path ${HOME}/data/VOC/VOCdevkit/VOC2012/instances_trainval.json \
    --image_dir ${HOME}/data/VOC/VOCdevkit/VOC2012/JPEGImages

# grounding dino inference on KITTI custom train data
CUDA_VISIBLE_DEVICES=0 python gdino_inference_kitti.py \
    -c configs/GroundingDINO_SwinB_cfg.py \
    -p checkpoints/groundingdino_swinb_cogcoor.pth \
    --anno_path ${HOME}/data/KITTI/train/soft_instances_train.json \
    --image_dir ${HOME}/data/KITTI/train/images


### CLIP crop embeddings (optional -- only needed for analysis/ablation.ipynb's
### CLIP-vs-GDino-embedding-graph section, COCO val only). Augments the parquet
### from the COCO val Grounding DINO block above with a clip_embedding column
### in place; label_spreading/build_matching_dataset.py picks it up automatically
### if present when the matched dataset is (re)built.

# python clip_embeddings.py --predictions_path predictions/gdino/coco/predictions.parquet \
#     --anno_path ${HOME}/data/coco/annotations/instances_val2017.json \
#     --image_dir ${HOME}/data/coco/val2017


### OWL-ViT / YOLO-World (only needed for analysis/ablation.ipynb's
### OVOD-ensemble section). Same scope as Grounding DINO -- full COCO val and
### train, no restriction at inference time (--save_path defaults already
### point at predictions/{owlvit,yolov8l-worldv2}/coco/predictions.parquet):

# CUDA_VISIBLE_DEVICES=0 python owlvit_inference_coco.py --image_dir ${HOME}/data/coco/val2017
# CUDA_VISIBLE_DEVICES=0 python owlvit_inference_coco.py --image_dir ${HOME}/data/coco/train2017 \
#     --save_path predictions/owlvit/coco_train/predictions.parquet

# CUDA_VISIBLE_DEVICES=0 python yoloworld_inference_coco.py --image_dir ${HOME}/data/coco/val2017
# CUDA_VISIBLE_DEVICES=0 python yoloworld_inference_coco.py --image_dir ${HOME}/data/coco/train2017 \
#     --save_path predictions/yolov8l-worldv2/coco_train/predictions.parquet

# ablation.ipynb's OVOD-ensemble section only calibrates/compares on the
# SoftTeacher fold=1/1%-budget labeled images (1182 of the 118k train images
# -- the only ones with ground truth to calibrate against), so filter the
# full-scope coco_train predictions above down to that subset:

# python filter_predictions_by_image_subset.py \
#     --predictions_path predictions/owlvit/coco_train/predictions.parquet \
#     --labels_json ${HOME}/SoftTeacher/data/coco/annotations/semi_supervised/instances_train2017.1@1.json \
#     --output_path predictions/owlvit/coco_train_fold1_1pct/predictions.parquet
# python filter_predictions_by_image_subset.py \
#     --predictions_path predictions/yolov8l-worldv2/coco_train/predictions.parquet \
#     --labels_json ${HOME}/SoftTeacher/data/coco/annotations/semi_supervised/instances_train2017.1@1.json \
#     --output_path predictions/yolov8l-worldv2/coco_train_fold1_1pct/predictions.parquet
