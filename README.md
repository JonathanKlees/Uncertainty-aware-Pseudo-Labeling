# Probabilistic Label Spreading for Object-Detection Annotation

Semi-supervised, uncertainty-aware annotation of object-detection datasets:
starting from a small labeled budget plus an open-vocabulary detector's
(Grounding DINO) raw detections, **Probabilistic Label Spreading (PLS)**
propagates soft (probabilistic) labels across a k-NN similarity graph built
in the detector's embedding space, and a **Bayesian blend** combines the
propagated estimate with the detector's own calibrated confidence. Evaluated
via Average Precision (AP) and Average Similarity (AS, class-agnostic
soft-label agreement; `AS_strict` in the code) across COCO, Cityscapes, KITTI
and Pascal VOC, plus downstream semi-supervised object detection (SSOD)
training, label-error detection and active querying.

This README documents the commands to reproduce every result reported in the
paper, and gives a runtime estimate wherever a step takes longer than a few
minutes.

Code for the paper [*Open-Vocabulary Models and Label Propagation for
Uncertainty-aware Pseudo-Labeling of Object Detection
Datasets*](https://arxiv.org/abs/2610.29788) (see [Citation](#citation)). The
soft ground truth comes from the soft-label benchmark of Penquitt et al.
(NeurIPS 2026 Evaluations & Datasets Track).

## Setup

Use a Conda environment (FAISS packages are only readily available there).

- **PLS core** (`label_spreading/src.py`) needs [pyamgx](https://github.com/NVIDIA/AMGX)
  (an older AMGX release may be required depending on your GPU — see the
  [AMGX releases](https://github.com/NVIDIA/AMGX/releases), then install
  pyamgx from a local checkout) and FAISS, currently satisfied by the
  `faiss-gpu-cu12` PyPI package (targets a specific CUDA version; if it's
  incompatible with your system, [build FAISS from
  source](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md)
  instead — an older [FAISS release](https://github.com/facebookresearch/faiss/releases)
  may be needed depending on your GPU).
- `requirements.lock.txt` is a full `pip freeze` of the working environment
  (Python 3.11, PyTorch 2.5.1+cu121) — install everything else from it.
  `pyamgx` is a local editable install and won't resolve on another machine
  from the file as-is; point it at your own pyamgx checkout instead. The
  `+cu121` PyTorch wheels need `--extra-index-url https://download.pytorch.org/whl/cu121`.
- The plotting code (`utils/plot_layout.py`) renders text with LaTeX, so a
  LaTeX installation with `cm-super` and `dvipng` is needed for the figures.
- `inference/setup.sh` sets up the separate environment used for Grounding
  DINO / OWL-ViT / YOLO-World inference (see `inference/README.md`).

## Repository layout

```
inference/             Open-vocabulary detector inference (Grounding DINO, OWL-ViT, YOLO-World)
label_spreading/       PLS core (src.py), evaluation.py, matching, the main AP/AS/baseline sweep, active querying
label_error_detection/ Label-error-detection benchmark (Bayesian PLS vs. Grounding DINO vs. random)
pseudo_labels/         COCO pseudo-label generation for the downstream SSOD (Soft Teacher) evaluation
analysis/              Result-gathering notebooks, figures and LaTeX tables for the paper
calibration/           Helpers to score OVOD detections against COCO GT (TP/FP per detection) for calibration
utils/                 Shared matplotlib/LaTeX plot styling (plot_layout.py)
data/                  Matched datasets, soft ground truth, download script (see "Data pipeline" below)
abstract_figure/       Paper abstract figure
method_figure/         Paper method figure (method_figure.py -> method_figure.{png,pdf})
```

## Data pipeline

Everything downstream reads from `data/matched_dataset/*_matched.parquet`
(Grounding DINO's detections IoU-matched against soft ground truth, per
dataset) and `data/soft_datasets/*.json` (the soft ground-truth label
distributions themselves). Both are gitignored (multi-GB parquet/JSON files);
regenerate them from scratch as follows.

**All commands below assume you `cd` into the directory shown first** — most
of these scripts default to relative paths (config files, output
directories) resolved against their own directory, not the repo root.

0. **Raw datasets** — the scripts expect the raw images and annotations under
   `~/data/`: `coco/{train2017,val2017,unlabeled2017,annotations}`,
   `VOC/VOCdevkit/VOC2012/`, `Cityscapes/{train,val}/` and `KITTI/train/`
   (see `inference/create_coco_style_annotations.ipynb` for the exact
   layout). The COCO annotations and SoftTeacher's semi-supervised splits
   (`annotations/semi_supervised/instances_train2017.<fold>@<percent>.json`)
   are read from a [SoftTeacher](https://github.com/microsoft/SoftTeacher)
   checkout at `~/SoftTeacher/`, which is also used for the SSOD training
   (§3); its `data/` directory is a symlink to `~/data/`, so both paths
   point at the same files. Then run `inference/create_coco_style_annotations.ipynb`
   once: it writes the COCO-format annotation files for VOC, Cityscapes and
   KITTI that inference, soft-GT harmonization and the label-error study
   read (`soft_instances_*.json` with the soft-label benchmark's boxes,
   `instances_*.json` with the official annotations; see
   `inference/README.md`).
1. **Soft ground truth** — `cd data && ./download_data.sh` downloads the raw
   crowd-sourced soft-GT files into `data/soft_datasets/`. Then
   `cd ../label_spreading && python build_soft_gt.py` harmonizes
   Cityscapes/KITTI/VOC's schema to match COCO's (a few seconds), writing the
   `soft_*_fixed.json` files.
2. **Detector inference** — `cd inference && ./inference.sh` runs Grounding
   DINO over all 4 datasets (COCO val/train/unlabeled, Cityscapes, Pascal
   VOC, KITTI). **Each block can take hours; COCO (118k train images)
   dominates wall-clock — run the whole script in tmux.** Output:
   `inference/predictions/gdino/<dataset>/predictions.parquet` (gitignored,
   ~132GB).

   More blocks are commented out in `inference.sh` (uncomment to use):
   - **CLIP crop embeddings** (`clip_embeddings.py`) — adds a
     `clip_embedding` column to the COCO val predictions in place. Needed
     before matching COCO val (step 3), and used by `analysis/ablation.ipynb`'s
     CLIP-vs-GDino-embedding-graph section.
   - **OWL-ViT / YOLO-World** — only needed for `ablation.ipynb`'s
     OVOD-ensemble section, which calibrates and compares on SoftTeacher's
     fold 1 / 1% labeled images (1,182 of the 118k train images). A further
     `filter_predictions_by_image_subset.py` step (also in `inference.sh`)
     filters the full `coco_train` predictions down to that subset.

   Total gitignored output across all detectors/datasets: ~309GB.
3. **Matching** (`cd label_spreading` first) — IoU-matches each dataset's
   raw detections against its ground truth:
   - **COCO val**: `build_matching_dataset.ipynb` (matches against the soft
     GT and, independently, the standard COCO GT; carries the CLIP crop
     embeddings through) → `coco_val_matched.parquet`.
   - **COCO train**: `build_matching_dataset_coco_train.ipynb` (no soft GT
     exists for COCO train, so it matches against the standard COCO GT only)
     → `coco_train_matched.parquet`, the input of the SSOD pseudo-label
     generation (§3).
   - **Cityscapes / KITTI / Pascal VOC**: `build_matching_dataset.py`, once
     per dataset. COCO uses the notebooks above because Grounding DINO's COCO
     labels are sparse COCO category ids rather than a dense class index.
   ```bash
   cd label_spreading
   python build_matching_dataset.py \
       --predictions_path ../inference/predictions/gdino/cityscapes/predictions.parquet \
       --soft_gt_path ../data/soft_datasets/soft_Cityscapes_trainval_fixed.json \
       --output_path ../data/matched_dataset/cityscapes_trainval_matched.parquet
   # repeat with --predictions_path .../gdino/{kitti,pascal}/predictions.parquet,
   # --soft_gt_path .../soft_{Kitti_2D_train,PascalVOC_2012_detection_trainval}_fixed.json,
   # --output_path .../data/matched_dataset/{kitti_train,pascal_voc_trainval}_matched.parquet
   ```
   Output: `data/matched_dataset/<dataset>_matched.parquet` (gitignored,
   ~13GB total).

## Reproducing the main results

### 1. AP/AS sweep + baseline comparison

`label_spreading/baseline_comparison.py` grid-searches PLS/GKR/kNN
hyperparameters and the Bayesian-blend κ, per dataset and annotation budget,
against the Baseline (calibrated Grounding DINO confidence) and Resort
(hard-label injection) references. It takes explicit required arguments
(`--dataset_path`, `--soft_gt_path`, `--results_dir`, `--run_name`) and its
`--amgx_config` default is relative, so run it from `label_spreading/`, once
per dataset:

```bash
cd label_spreading
python baseline_comparison.py \
    --dataset_path ../data/matched_dataset/coco_val_matched.parquet \
    --soft_gt_path ../data/soft_datasets/soft_COCO_2017_val.json \
    --results_dir ../analysis/experiments/coco_val_baseline_comp_simple/grid_search \
    --run_name coco_val_baseline_comp_simple
# repeat for cityscapes_trainval / kitti_train / voc_trainval (the script's
# docstring lists the parquet, soft-GT file and run name for all four)
```

Outputs `analysis/experiments/<run_name>/grid_search/*.csv`, consumed by the
analysis notebooks. The PLS grid is 150 combos (2 `k` × 5 `bandwidth_scale`
× 15 `alpha`) per (dataset, budget), over 7 budgets (1, 2, 5, 10, 25, 50,
100%). COCO val (~86k detections) took **~6 hours**; plan on several hours
per dataset and run it in tmux.

All reported AP/AS values of this benchmark are the best grid point per
method and budget, evaluated on the full dataset; this benchmark has no
held-out split for hyperparameter selection (unlike the SSOD pseudo-labels
in §3 and the label-error detection in §5).

### 2. Hyperparameter study / ablations

`analysis/ablation.ipynb`, four sections: (1) OVOD-ensemble precision/recall/F1
across confidence thresholds — needs the OWL-ViT/YOLO-World predictions from
the data pipeline's `inference.sh` step; (2) AS/AP for PLS vs. GKR vs. kNN
across all four datasets, from §1's grid results; (3) CLIP-embedding vs.
GDino-embedding similarity graph on COCO val (fixed `k=10`, `alpha=0.9`, so
its absolute numbers are not comparable to §1's tuned results); (4) the
paper's ablation table (`tables/ablation_ovod_clip.tex`). Runs in minutes once
its inputs exist, apart from section 3's PLS runs and evaluations.

`label_spreading/gdino_score_analysis.ipynb` sweeps the Grounding DINO
score threshold applied before spreading (11 thresholds × 15 `alpha` × 3
budgets, `k=20`, COCO val), feeding the `PLS_gdino_score_heatmap_*.pdf`
plots in `analysis/annotation_benchmark_results.ipynb`. Long-running; run it
unattended.

### 3. Pseudo-label generation for COCO / SSOD downstream evaluation

Grounding DINO-only pseudo-labels (calibrated confidences, thresholded) come
from `pseudo_labels/grounding_dino_pseudo_labels.ipynb`. Bayesian-PLS
pseudo-labels for SoftTeacher's semi-supervised COCO splits (5 folds × {1, 5,
10}% budget) plus a 100%-budget setting come from:

```bash
python pseudo_labels/generate_pls_pseudo_labels_sweep.py   # 15 fold/budget combos
python pseudo_labels/generate_pls_pseudo_labels_100pct.py  # 100% budget
```

**Long unattended jobs — run in tmux.** Both are checkpointed per combo
(safe to resume). Runtime not logged; each combo runs its own
hyperparameter search over the ~2M-detection COCO train graph, so expect the
full sweep to take multiple hours to days depending on GPU. Output:
`pseudo_labels/{grounding_dino,bayesian_pls}_pseudo_labels/` (the pseudo-label
files themselves are gitignored, ~10GB combined), consumed by SoftTeacher
training (external repo), whose logs `analysis/SSOD_benchmark_results.ipynb`
plots.

The hyperparameters (k, alpha, κ) and the pseudo-label threshold of each
combo are selected on an 80/20 image-level split of that combo's own labeled
images (`SEARCH_VAL_FRAC`), never on the unlabeled images that receive the
pseudo-labels. `pseudo_labels/compute_train_test_f1_gap.py` shows why:
F1 measured on the labeled images themselves is far too optimistic compared
to the unlabeled population (e.g. fold 1 @ 1% budget: train F1 = 0.998, test
F1 = 0.784; all 15 combos in
`pseudo_labels/bayesian_pls_pseudo_labels/train_test_f1_gap_summary.csv`).
The selected values are tracked in `pls_hp_search_*.json`; fold 1 @ 10% was
generated before the sweep script wrote these files, so its values are
recorded in `compute_train_test_f1_gap.py` instead.

### 4. Active querying

```bash
cd label_spreading   # AMGX_CONFIG default is "./configs/...", a relative path
python active_querying.py
```

Single-instance active querying on COCO val (PLS re-solved after every
query, evaluated every 100 queries, up to a 10% budget) for the strategies
random, entropy (Aleatoric), trials (Epistemic), mixture (Aleatoric &
Epistemic) and Graph-aware Uncertainty. One strategy takes **~4.5–5 hours**
(86 evaluated checkpoints), all strategies **~30 hours** — run it in tmux.
The script also stores each strategy's query order, which the spread metrics
and UMAP plots use. Output feeds `analysis/active_querying.ipynb`
(trajectory, query-spread metrics and UMAP visualizations; runs in minutes
given the script's output, plus a one-off UMAP fit that is cached).
`label_spreading/reconstruct_query_history.py` recovers the query order of
runs made without it.

### 5. Label error detection

Label errors are defined in `label_error_detection/define_label_errors.ipynb`
(output: `label_error_detection/label_errors/`, gitignored — regenerates in
seconds): objects in the hard-thresholded soft GT that are missing,
misaligned or misclassified in the datasets' original annotations. A
hyperparameter search (on an 80/20 image-level split) plus final evaluation
covers all 4 datasets:

```bash
cd label_error_detection   # AMGX_CONFIG/logs/label_errors paths are all relative
python parameter_sweep.py         # all 4 datasets, checkpointed
cd ../analysis              # sys.path/RESULTS_PICKLE/tables/plots paths are all relative
python label_error_detection_results.py   # final LaTeX tables + PR-curve PDFs from the checkpoint, no re-solving
```

The sweep searches 1,200 (k, bandwidth, alpha) combos (1,080 for KITTI and
Pascal VOC, where `bandwidth_scale=0.01` is excluded), each with 16 κ values.
It runs two Grounding DINO score thresholds (0.1 and 0.01); only 0.01 is
reported. `parameter_sweep.py`'s checkpoint pickle and per-combo
search-surface logs land in `label_error_detection/logs/` (gitignored); the
reported tables and plots land in `analysis/tables/` and `analysis/plots/`.

**Long unattended job — run it as a plain `python` process in tmux**, not
via a Jupyter kernel. Logged runtimes per dataset: COCO 1.6h/5.7h, Cityscapes
0.5h/1.5h, KITTI 0.5h/2.6h, Pascal VOC 0.3h/4.7h (thr=0.1 / thr=0.01) —
**~17.4 hours total**.

### 6. Analysis / figures

All in `analysis/`, run from that directory:

- `annotation_benchmark_results.ipynb`: AP/AS vs. budget, the summary table
  (`tables/as_ap_comparison.tex`), the AS ceiling, the
  score-to-annotation precision/recall breakdown, the annotation-cost
  estimate (`tables/annotation_cost.tex`, from the answer times in the
  soft-GT files), hyperparameter sweeps and AS/AP curves at fixed IoU=0.5.
- `visual_soft_label_examples.ipynb` (COCO) and `visual_soft_label_examples.py`
  (`--dataset {voc_trainval,cityscapes_trainval,kitti_train}`, the other
  datasets) render qualitative soft-label examples into
  `plots/examples/`; `build_visual_examples_latex.py` assembles the selected
  ones into `tables/visual_examples_{agreement,disagreement,uncertain_gt,other_datasets}.tex`
  (image paths there point at the paper's `imgs/examples/`).
- `label_error_detection_results.py` (see §5), `active_querying.ipynb`
  (see §4), `ablation.ipynb` (see §2).

The intermediate results these notebooks read (grid-search results, AS
ceilings, answer times, active-querying results, query histories and the UMAP
embedding, ablation results; ~11MB) are tracked in `analysis/experiments/`, so
none of the expensive upstream steps has to be rerun. Most notebooks still read
the matched datasets and soft ground truth from `data/` (see "Data pipeline"):
`annotation_benchmark_results.ipynb` and `active_querying.ipynb` then run in
minutes; `ablation.ipynb` and the visual examples additionally need the
detector predictions and images, and `label_error_detection_results.py` needs
`parameter_sweep.py`'s checkpoint. `SSOD_benchmark_results.ipynb` runs from a
fresh clone as is. Each notebook's markdown cells document which upstream
result it reads.

`SSOD_benchmark_results.ipynb` (downstream mAP bar chart and training curves)
reads the SoftTeacher training logs (`*.log.json`, one directory per training
run) tracked in `analysis/ssod_logs/`. Producing them requires the Faster
R-CNN training via [SoftTeacher](https://github.com/microsoft/SoftTeacher),
which isn't scripted here: use §3's pseudo-labels as SoftTeacher's supervision
source and follow SoftTeacher's own training instructions. **Full SSOD
evaluation there takes ~9 days on 8 GPUs.**

## Data not tracked in git

The following outputs are too large for git and are gitignored (see
`.gitignore`); see the relevant section above for the command that
regenerates each.

| Path | Contents | Approx. size |
|---|---|---|
| `inference/predictions/` | Raw per-detector, per-dataset detections | ~309GB |
| `inference/checkpoints/`, `inference/configs/` | Detector weights and config (downloaded by `setup.sh`) | ~1GB |
| `data/matched_dataset/` | Detections IoU-matched to soft GT | ~13GB |
| `data/soft_datasets/` | Soft ground-truth label distributions (`data/download_data.sh`) | ~1.3GB |
| `pseudo_labels/*/pseudo_labels_*.json` | Grounding DINO-only and Bayesian-PLS SSOD pseudo-labels | ~7.4GB |
| `pseudo_labels/*/calibration_inputs_*.json` | Cached per-detection calibration inputs | ~2.8GB |
| `analysis/tables/*.npz` | Raw per-combo curve-array dumps behind the label-error-detection plots | ~170MB |
| `label_error_detection/logs/` | `parameter_sweep.py`'s checkpoint pickle + full search-surface logs, read by `label_error_detection_results.py` | ~220MB |
| `label_error_detection/label_errors/` | Label-error ground truth (regenerates in seconds) | ~10MB |

## Citation

If you use this code, please cite our paper:

```bibtex
@misc{klees2026openvocabularymodelslabelpropagation,
      title={{Open-Vocabulary Models and Label Propagation for Uncertainty-aware Pseudo-Labeling of Object Detection Datasets}},
      author={Jonathan Klees and Timo Heller and Matthias Rottmann},
      year={2026},
      eprint={2610.29788},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.29788},
}
```

and, if you use the soft ground truth, the soft-label benchmark:

```bibtex
@inproceedings{penquitt2026object,
  title     = {{Object Detection Benchmarks are Incomplete: The Role of Label Errors and Annotation Uncertainty}},
  author    = {Penquitt, Sarina and Klees, Jonathan and van Betteray, Antonia and Jashnieh, Parssa and Stehr, Peter and Rottmann, Matthias and Schmarje, Lars},
  booktitle = {Advances in Neural Information Processing Systems Evaluations \& Datasets Track},
  year      = {2026},
}
```

## License

The code in this repository is released under the [MIT License](LICENSE).

This license covers only the code written for this project. The repository
builds on pre-existing software, models and datasets that come with their own
licenses, which you have to respect when using them, in particular:

- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) (Apache-2.0), including its
  pretrained weights, and [Detectron2](https://github.com/facebookresearch/detectron2) (Apache-2.0),
- [OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit) via Hugging Face Transformers (Apache-2.0),
  [YOLO-World](https://docs.ultralytics.com/models/yolo-world/) via Ultralytics (AGPL-3.0) and
  [CLIP](https://github.com/openai/CLIP) (MIT),
- [SoftTeacher](https://github.com/microsoft/SoftTeacher) (MIT), used for the SSOD training,
- [AMGX](https://github.com/NVIDIA/AMGX) / [pyamgx](https://github.com/shwina/pyamgx) and
  [FAISS](https://github.com/facebookresearch/faiss) (MIT),
- the label-error matching and AP methodology in `label_error_detection/label_error_utils.py`,
  ported from [Uncertainty-aware-Object-Detection](https://github.com/JonathanKlees/Uncertainty-aware-Object-Detection),
- the datasets: [COCO](https://cocodataset.org/#termsofuse),
  [Cityscapes](https://www.cityscapes-dataset.com/license/),
  [KITTI](https://www.cvlibs.net/datasets/kitti/) and
  [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/), each under its own terms
  (Cityscapes and KITTI are restricted to non-commercial use), and the soft
  labels of the soft-label benchmark (see its Dataverse page).

