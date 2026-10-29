"""Honest hyperparameter search + final evaluation for the label-error-detection
benchmark: PLS+Bayes vs. GDino baseline on all 4 datasets. Runs both score
thresholds in SCORE_THRESHOLDS (0.1 and 0.01); only thr=0.01 is reported
(analysis/label_error_detection_results.py). Label-error ground truth is the
reference-consistent definition (`define_label_errors.ipynb`) -- fixing two
discrepancies found vs. `Uncertainty-aware-Object-Detection`'s own
`helpers_legt.py::extract_label_errors` (dropping unmatched-but-IoU>=0.5 boxes;
adding classification-type errors, matched pairs with IoU>=0.7 and a different
class).

Logged total runtime ~17.4h (thr 0.1 / 0.01: COCO 1.6h / 5.7h, Cityscapes
0.5h / 1.5h, KITTI 0.5h / 2.6h, VOC 0.3h / 4.7h). Run as a plain python
process, e.g. in tmux, since long Jupyter-kernel runs proved unreliable.

Checkpointed per (score_thr, dataset_key), so an interrupted run resumes
where it stopped.

`analysis/label_error_detection_results.py` builds the paper-ready
comparison table, hyperparameter table and PR-curve plots directly from
this script's checkpoint pickle afterward -- no re-solving.
"""
import os
import sys
import json
import time
import pickle

import numpy as np
import pandas as pd
import pyamgx
from sklearn.metrics import average_precision_score
from sklearn.isotonic import IsotonicRegression

sys.path.append("../label_spreading")
sys.path.append("../utils")
from src import spreading_conventional, set_graph
from label_error_utils import (original_gt_to_hard_boxes, match_detections_to_gt, compute_pr_curve,
                                compute_ap, random_baseline, apply_nms)

os.makedirs("logs", exist_ok=True)

SCORE_THRESHOLDS = [0.1, 0.01]
NMS_IOU_THRESH = 0.6
MATCH_IOU_THRESH = 0.5
LE_IOU_THRESH = 0.1
AMGX_CONFIG = "../label_spreading/configs/FGMRES_AGGREGATION.json"
RANDOM_BASELINE_SEED = 0

SEARCH_VAL_FRAC = 0.2
SEARCH_SEED = 123
K_GRID = [10, 20, 50, 100, 150, 200]
ALPHA_GRID = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
BANDWIDTH_GRID = [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, 2.0]
KAPPA_GRID = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, np.inf]

DATASETS = {
    "coco_val": dict(
        display_name="COCO",
        original_gt="/home/klees/SoftTeacher/data/coco/annotations/instances_val2017.json",
        matched_path="../data/matched_dataset/coco_val_matched.parquet",
    ),
    "cityscapes_trainval": dict(
        display_name="Cityscapes",
        original_gt="/home/klees/data/Cityscapes/trainval/instances_trainval.json",
        matched_path="../data/matched_dataset/cityscapes_trainval_matched.parquet",
    ),
    "kitti_train": dict(
        display_name="KITTI",
        original_gt="/home/klees/data/KITTI/train/instances_train.json",
        matched_path="../data/matched_dataset/kitti_train_matched.parquet",
    ),
    "voc_trainval": dict(
        display_name="Pascal VOC",
        original_gt="/home/klees/data/VOC/VOCdevkit/VOC2012/instances_trainval.json",
        matched_path="../data/matched_dataset/pascal_voc_trainval_matched.parquet",
    ),
}

# bandwidth_scale=0.01 yields unstable solves on the larger KITTI/VOC graphs, so it
# is excluded there (6 k x 9 bandwidth x 20 alpha = 1,080 PLS combos instead of 1,200).
UNSAFE_BANDWIDTHS_BY_DATASET = {"kitti_train": {0.01}, "voc_trainval": {0.01}}

RESULTS_PICKLE = "logs/parameter_sweep_all_results.pkl"


def build_graphs(det_df, n_data):
    S_by_k_bw = {}
    for k in K_GRID:
        for bw in BANDWIDTH_GRID:
            S_by_k_bw[(k, bw)], _, _, _ = set_graph(k, det_df, "gdino_embedding", n_data=n_data, bandwidth_scale=bw)
    return S_by_k_bw


def get_F_obj(det_df, classes, S, alpha, tp_indices):
    try:
        _, F_object, _, _, _ = spreading_conventional(alpha, S, det_df, "soft_label", "is_object_original",
                                                        classes, AMGX_CONFIG, tp_indices)
    except pyamgx.AMGXError as e:
        print(f"    !! AMGXError at alpha={alpha} -- skipping this combo ({e})", flush=True)
        return None, None
    return F_object[:, 1], F_object[:, 0]


def bayes_score(F_obj_1, F_obj_0, prior_obj, kappa):
    if np.isinf(kappa):
        return prior_obj
    unnorm_1 = kappa * prior_obj + F_obj_1
    unnorm_0 = kappa * (1 - prior_obj) + F_obj_0
    denom = unnorm_0 + unnorm_1
    return np.divide(unnorm_1, denom, out=np.full_like(unnorm_1, 0.5), where=denom > 0)


def f1_from_pr(precision, recall):
    p = np.asarray(precision)
    r = np.asarray(recall)
    f1 = np.where((p + r) > 0, 2 * p * r / (p + r + 1e-12), 0.0)
    idx = int(np.argmax(f1))
    return float(f1[idx]), float(p[idx]), float(r[idx])


def honest_search(det_df, classes, n_data, matched_original, raw_score, S_by_k_bw, dataset_key, search_log_path):
    """Also writes every evaluated (k, bandwidth, alpha, kappa) combo's held-out
    val_ap to `search_log_path` (not just the running best), so the full search
    surface can be inspected afterward (e.g. whether a winner sits at a grid edge)."""
    all_image_ids = np.array(sorted(det_df["image_id"].unique()))
    rng = np.random.default_rng(SEARCH_SEED)
    perm = rng.permutation(len(all_image_ids))
    n_val = max(1, int(len(all_image_ids) * SEARCH_VAL_FRAC))
    val_image_ids = set(all_image_ids[perm[:n_val]].tolist())
    sup_image_ids = set(all_image_ids[perm[n_val:]].tolist())

    sup_mask = det_df["image_id"].isin(sup_image_ids).to_numpy()
    val_mask = det_df["image_id"].isin(val_image_ids).to_numpy()

    search_tp = np.where(matched_original & sup_mask)[0]
    val_indices = np.where(val_mask)[0]
    y_val = matched_original[val_indices]

    ir_search = IsotonicRegression(out_of_bounds="clip")
    ir_search.fit(raw_score[sup_mask], matched_original[sup_mask].astype(float))
    prior_search = ir_search.predict(raw_score)

    unsafe_bw = UNSAFE_BANDWIDTHS_BY_DATASET.get(dataset_key, set())

    best = {"val_ap": -1.0}
    all_rows = []
    n_skipped = 0
    n_unsafe_skipped = 0
    n_done = 0
    total = (len(S_by_k_bw) - sum(1 for (k, bw) in S_by_k_bw if bw in unsafe_bw)) * len(ALPHA_GRID)
    for (k, bw), S in S_by_k_bw.items():
        if bw in unsafe_bw:
            n_unsafe_skipped += len(ALPHA_GRID)
            continue
        for alpha in ALPHA_GRID:
            F_obj_1, F_obj_0 = get_F_obj(det_df, classes, S, alpha, search_tp)
            n_done += 1
            if F_obj_1 is None:
                n_skipped += 1
                continue
            for kappa in KAPPA_GRID:
                score = bayes_score(F_obj_1, F_obj_0, prior_search, kappa)
                ap = average_precision_score(y_val, score[val_indices])
                all_rows.append(dict(k=k, alpha=alpha, bandwidth_scale=bw, kappa=kappa, val_ap=ap))
                if ap > best["val_ap"]:
                    best = dict(val_ap=ap, k=k, alpha=alpha, bandwidth_scale=bw, kappa=kappa)
            if n_done % 20 == 0:
                print(f"    {n_done}/{total} (k, bandwidth, alpha) combos searched, "
                      f"current best val_ap={best['val_ap']:.4f} "
                      f"(k={best.get('k')}, alpha={best.get('alpha')}, bw={best.get('bandwidth_scale')})", flush=True)
                # flushed every 20 combos so an interrupted search leaves a partial log
                pd.DataFrame(all_rows).to_csv(search_log_path, index=False)
    if n_skipped:
        print(f"  ({n_skipped} (k, bandwidth, alpha) combos skipped due to AMGXError)", flush=True)
    if n_unsafe_skipped:
        print(f"  ({n_unsafe_skipped} combos pre-emptively skipped: bandwidth in {unsafe_bw} "
              f"is known-fatal on {dataset_key})", flush=True)
    pd.DataFrame(all_rows).to_csv(search_log_path, index=False)
    print(f"  full search log ({len(all_rows):,} rows) -> {search_log_path}", flush=True)
    if best["val_ap"] < 0:
        raise RuntimeError("honest_search: every single combo failed -- not a one-off edge case, needs investigation")
    return best


def main():
    all_results = {}
    if os.path.exists(RESULTS_PICKLE):
        with open(RESULTS_PICKLE, "rb") as f:
            all_results = pickle.load(f)
        n_done = sum(len(v) for v in all_results.values())
        print(f"Resuming from checkpoint: {RESULTS_PICKLE} already has {n_done} dataset results", flush=True)

    for score_thr in SCORE_THRESHOLDS:
        all_results.setdefault(score_thr, {})
        for dataset_key, cfg in DATASETS.items():
            if dataset_key in all_results[score_thr]:
                print(f"\n{cfg['display_name']} (SCORE_THR={score_thr}) already done, skipping", flush=True)
                continue

            t_dataset = time.time()
            print(f"\n{'=' * 70}\n{cfg['display_name']}  (SCORE_THR={score_thr})\n{'=' * 70}", flush=True)

            with open(cfg["original_gt"]) as f:
                original_gt_json = json.load(f)
            original_by_image, _ = original_gt_to_hard_boxes(original_gt_json)

            det_df = pd.read_parquet(cfg["matched_path"],
                                      columns=["image_id", "gdino_bbox", "gdino_score", "gdino_embedding", "soft_label", "label"])
            det_df = det_df[det_df["gdino_score"] >= score_thr].reset_index(drop=True)
            n_before_nms = len(det_df)
            det_df = apply_nms(det_df, iou_threshold=NMS_IOU_THRESH)
            n_data = len(det_df)
            print(f"n_data={n_data:,} (before NMS: {n_before_nms:,}, -{1 - n_data / n_before_nms:.1%})", flush=True)

            matched_original = match_detections_to_gt(det_df, original_by_image, iou_threshold=MATCH_IOU_THRESH)
            det_df["is_object_original"] = matched_original.astype(float)
            det_df["soft_label"] = list(np.array([v for v in det_df["soft_label"].values]))
            classes = sorted(det_df["label"].dropna().unique())
            raw_score = det_df["gdino_score"].to_numpy()
            print(f"confirmed TPs={matched_original.sum():,}  originally-FP={int((~matched_original).sum()):,}", flush=True)

            t0 = time.time()
            S_by_k_bw = build_graphs(det_df, n_data)
            print(f"graphs built [{time.time() - t0:.1f}s]", flush=True)

            t0 = time.time()
            search_log_path = f"logs/parameter_sweep_search_log_{dataset_key}_thr{score_thr}.csv"
            best = honest_search(det_df, classes, n_data, matched_original, raw_score, S_by_k_bw,
                                  dataset_key, search_log_path)
            edge = " (= pure PLS)" if best["kappa"] == 0.0 else \
                   " (= pure calibrated GDino)" if np.isinf(best["kappa"]) else ""
            print(f"  [unlabeled_fp best] {best}{edge}", flush=True)
            print(f"honest search done [{time.time() - t0:.1f}s]", flush=True)

            with open(f"label_errors/{dataset_key}_label_errors.json") as f:
                label_errors = json.load(f)

            all_tp = np.where(matched_original)[0]
            fp_mask = ~matched_original
            originally_fp_df = det_df[fp_mask].reset_index(drop=True)

            ir_full = IsotonicRegression(out_of_bounds="clip")
            ir_full.fit(raw_score, matched_original.astype(float))
            prior_full = ir_full.predict(raw_score)

            def build_proposals(score):
                return [{"image_id": r.image_id, "bbox": r.gdino_bbox, "score": float(s)}
                        for r, s in zip(originally_fp_df.itertuples(), score[fp_mask])]

            def evaluate_score(score, method_name):
                proposals = build_proposals(score)
                precision, recall, scores = compute_pr_curve(proposals, label_errors, iou_thresh=LE_IOU_THRESH)
                ap = compute_ap(precision, recall)
                achieved_recall = float(recall[-1]) if len(recall) else 0.0
                f1, f1_prec, f1_rec = f1_from_pr(precision, recall)
                print(f"  [{method_name}] FINAL: AP={ap:.4f}  best_F1={f1:.4f} (P={f1_prec:.4f}, R={f1_rec:.4f})  "
                      f"achieved_recall={achieved_recall:.4f}", flush=True)
                return dict(precision=precision, recall=recall, ap=ap, achieved_recall=achieved_recall,
                            f1=f1, f1_precision=f1_prec, f1_recall=f1_rec)

            dataset_results = {}
            dataset_results["gdino"] = evaluate_score(raw_score, "gdino")

            random_proposals = build_proposals(raw_score)
            rand_res = random_baseline(random_proposals, label_errors, iou_thresh=LE_IOU_THRESH, seed=RANDOM_BASELINE_SEED)
            print(f"  [random] FINAL: AP={rand_res['ap']:.4f}  F1={rand_res['f1']:.4f}  "
                  f"achieved_recall={rand_res['achieved_recall']:.4f}", flush=True)
            dataset_results["random"] = rand_res

            # blend final run
            t0 = time.time()
            S = S_by_k_bw[(best["k"], best["bandwidth_scale"])]
            F_obj_1, F_obj_0 = get_F_obj(det_df, classes, S, best["alpha"], all_tp)
            if F_obj_1 is None:
                raise RuntimeError(f"Final run solve failed for the WINNING combo {best} -- "
                                    f"this combo succeeded during honest_search, so this is unexpected")
            score = bayes_score(F_obj_1, F_obj_0, prior_full, best["kappa"])
            dataset_results["unlabeled_fp"] = {**evaluate_score(score, "unlabeled_fp"), **best}
            print(f"  unlabeled_fp done [{time.time() - t0:.1f}s]", flush=True)

            all_results[score_thr][dataset_key] = dataset_results
            print(f"{cfg['display_name']} (SCORE_THR={score_thr}) done [{time.time() - t_dataset:.0f}s]", flush=True)

            with open(RESULTS_PICKLE, "wb") as f:
                pickle.dump(all_results, f)
            print(f"  checkpoint saved -> {RESULTS_PICKLE}", flush=True)

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
