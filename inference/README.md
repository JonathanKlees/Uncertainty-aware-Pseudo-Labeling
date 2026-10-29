# Inference

Open-vocabulary object detector (OVOD) inference: Grounding DINO on all four
datasets, plus OWL-ViT / YOLO-World on COCO for the OVOD-ensemble ablation.

## Setup

```bash
./setup.sh
```

`setup.sh` builds the separate conda environment `open_vocabs` used for
inference (Python 3.10, CUDA 11.8 toolchain, PyTorch, Grounding DINO,
Detectron2, CLIP, Ultralytics). Note what it does outside this directory:

- removes and recreates the conda environment `open_vocabs` if it already exists,
- clones Grounding DINO to `~/GroundingDINO` (an existing checkout there is
  reset with `git checkout -- .`, since its CUDA sources get patched for
  PyTorch >= 2.1) and Detectron2 to `~/detectron2`.

Inside `inference/` it downloads the Grounding DINO Swin-B config to
`configs/GroundingDINO_SwinB_cfg.py` and its weights to
`checkpoints/groundingdino_swinb_cogcoor.pth`, and creates
`YW_OWL_Model_Cache_dir/` for the OWL-ViT / YOLO-World weights. All three
directories are gitignored.

## COCO-format annotation files

The inference scripts read each dataset's image list from a COCO-format
annotation JSON. COCO ships its own; for the other three datasets,
`create_coco_style_annotations.ipynb` creates them (see the notebook for the
expected directory layout under `~/data/`):

| Dataset | File | Boxes | Used by |
|---|---|---|---|
| Pascal VOC | `VOC/VOCdevkit/VOC2012/instances_trainval.json` | official VOC annotations | inference, soft-GT harmonization, label-error detection |
| Cityscapes | `Cityscapes/trainval/soft_instances_trainval.json` | boxes of the soft-label benchmark | inference, soft-GT harmonization |
| Cityscapes | `Cityscapes/trainval/instances_trainval.json` | official Cityscapes annotations | label-error detection |
| KITTI | `KITTI/train/soft_instances_train.json` | boxes of the soft-label benchmark | inference, soft-GT harmonization |
| KITTI | `KITTI/train/instances_train.json` | official KITTI annotations | label-error detection |

## Running inference

```bash
./inference.sh
```

Run from this directory. It runs Grounding DINO on COCO val/train/unlabeled,
Cityscapes trainval, Pascal VOC trainval and KITTI train, writing
`predictions/gdino/<dataset>/predictions.parquet`. Each block can take
hours (COCO train dominates), so run it in tmux. The optional blocks for
CLIP crop embeddings and OWL-ViT / YOLO-World (only needed for
`analysis/ablation.ipynb`) are commented out; uncomment to use.

## Contents

- `setup.sh`: inference environment setup
- `inference.sh`: runs all inference steps
- `gdino_inference_{coco,pascal,cityscapes,kitti}.py`: Grounding DINO inference, one script per dataset
- `owlvit_inference_coco.py`, `yoloworld_inference_coco.py`: OWL-ViT / YOLO-World inference (COCO only)
- `clip_embeddings.py`: adds CLIP crop embeddings to the COCO val predictions (CLIP-graph ablation)
- `filter_predictions_by_image_subset.py`: restricts COCO train predictions to a SoftTeacher labeled split
- `create_coco_style_annotations.ipynb`: COCO-format annotation files for VOC, Cityscapes and KITTI
- `requirements_inference.txt`: pip requirements of the inference environment
