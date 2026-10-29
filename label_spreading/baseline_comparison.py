"""Grid search comparing PLS (label spreading), GKR, kNN regression, Bayes
(PLS blended with calibrated GDino), the calibrated-GDino Baseline and Resort
across annotation budgets, on one dataset's matched-detections parquet + soft-GT
json. Writes f"{run_name}_grid_search_{results,best}.csv" to --results_dir.

Inputs per dataset (paths relative to this directory):
    COCO val     ../data/matched_dataset/coco_val_matched.parquet
                 ../data/soft_datasets/soft_COCO_2017_val.json
                 run name coco_val_baseline_comp_simple
    Cityscapes   ../data/matched_dataset/cityscapes_trainval_matched.parquet
                 ../data/soft_datasets/soft_Cityscapes_trainval_fixed.json
                 run name cityscapes_trainval_baseline_comp_simple
    KITTI        ../data/matched_dataset/kitti_train_matched.parquet
                 ../data/soft_datasets/soft_Kitti_2D_train_fixed.json
                 run name kitti_train_baseline_comp_simple
    Pascal VOC   ../data/matched_dataset/pascal_voc_trainval_matched.parquet
                 ../data/soft_datasets/soft_PascalVOC_2012_detection_trainval_fixed.json
                 run name voc_trainval_baseline_comp_simple

This is a long run (COCO val's took ~6h at ~86k detections after the score
threshold) -- run it via tmux, not interactively.

Usage (run from this directory -- --amgx_config defaults to ./configs/..., a
relative path):
    python baseline_comparison.py \\
        --dataset_path ../data/matched_dataset/coco_val_matched.parquet \\
        --soft_gt_path ../data/soft_datasets/soft_COCO_2017_val.json \\
        --results_dir ../analysis/experiments/coco_val_baseline_comp_simple/grid_search \\
        --run_name coco_val_baseline_comp_simple

    # smoke test with a tiny grid, to validate end-to-end before the full run:
    python baseline_comparison.py --quick \\
        --dataset_path ../data/matched_dataset/pascal_voc_trainval_matched.parquet \\
        --soft_gt_path ../data/soft_datasets/soft_PascalVOC_2012_detection_trainval_fixed.json \\
        --results_dir ../analysis/experiments/voc_trainval_baseline_comp_simple/grid_search \\
        --run_name voc_trainval_baseline_comp_simple

There is no --dataset flag -- --run_name is what names the output files;
--results_dir/--run_name are both required, no defaults.
"""
import os
import io
import sys
import argparse
import contextlib
import time
import multiprocessing

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Forking a process that already holds a CUDA context (from PLS's AMGX solve,
# which runs before each Parallel(...) call below) is unsafe and can deadlock;
# "spawn" avoids inheriting the parent's CUDA state. The spawned worker pool is
# reused across batches/budgets, so the startup cost is paid once.
multiprocessing.set_start_method("spawn", force=True)
from sklearn.linear_model import LogisticRegression

from pycocotools.cocoeval import COCOeval

from src import (
    label_spreading_object_conventional,
    knn_regression_object,
    gkr_object,
    _faiss_labeled_neighbors,
    _apply_object_conditioning,
)
from evaluation import evaluate_result, _load_hard_gt_from_soft

pd.set_option("display.max_columns", None)

TP_TIE_BREAK_EPS = 1e-6

# ── Full hyperparameter grids ──────────────────────────────────────────────────
# PLS: 2 k x 5 bandwidth_scale x 15 alpha = 150 combos per budget; bandwidth_scale
# scales the Gaussian-kernel bandwidth mu (mean squared kNN distance) in set_graph.
PLS_K_GRID_FULL = [10, 20]
PLS_ALPHA_GRID_FULL = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999]
PLS_BANDWIDTH_GRID_FULL = [0.1, 0.2, 0.5, 1.0, 2.0]   # x the mean squared kNN distance; below ~0.05-0.1 the graph goes numerically unstable
GKR_GAMMA_GRID_FULL = np.logspace(-6, 6, 60)
KNN_K_GRID_FULL = [2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 75, 100, 200, 300, 500]
BAYES_KAPPA_GRID_FULL = [0.1, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1e3, 2e3, 5e3, 1e4, 1e5]
BUDGETS_FULL = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]

# ── Tiny grids for --quick smoke testing (validate the pipeline runs end-to-end
# on a new dataset in ~a minute instead of hours, before committing to the real run) ──
PLS_K_GRID_QUICK = [20]
PLS_ALPHA_GRID_QUICK = [0.1, 0.5, 0.9]
PLS_BANDWIDTH_GRID_QUICK = [1.0]
GKR_GAMMA_GRID_QUICK = [1.0, 10.0]
KNN_K_GRID_QUICK = [10, 50]
BAYES_KAPPA_GRID_QUICK = [1.0, 100.0]
BUDGETS_QUICK = [0.01, 0.10]


def _category_aware_tp(dataset_in, soft_gt_path, soft_label_thresh, iou=0.5):
    """Resort baseline's ground truth: COCOeval's own per-category TP/FP matching,
    read directly off its evalImgs. Unlike `gdino_match` (a category-agnostic box
    match), a detection only counts as TP here if its predicted category matches."""
    coco_gt = _load_hard_gt_from_soft(soft_gt_path, soft_label_thresh)
    name_to_id = {cat["name"]: cat["id"] for cat in coco_gt.dataset["categories"]}
    cat_ids = dataset_in["gdino_label"].map(name_to_id)
    valid = cat_ids.notna().values

    boxes_xyxy = np.vstack(dataset_in.loc[valid, "gdino_bbox"].to_numpy()).astype(float)
    boxes_xywh = boxes_xyxy.copy()
    boxes_xywh[:, 2] -= boxes_xywh[:, 0]
    boxes_xywh[:, 3] -= boxes_xywh[:, 1]

    detections = [
        {"image_id": int(iid), "category_id": int(cid), "bbox": box.tolist(), "score": float(s)}
        for iid, cid, box, s in zip(
            dataset_in.loc[valid, "image_id"], cat_ids[valid].astype(int),
            boxes_xywh, dataset_in.loc[valid, "gdino_score"])
    ]
    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(detections)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()

    iou_idx = int(np.where(np.isclose(ev.params.iouThrs, iou))[0][0])
    all_area = ev.params.areaRng[0]
    tp_valid = np.zeros(int(valid.sum()), dtype=bool)
    for e in ev.evalImgs:
        if e is None or e["aRng"] != all_area:
            continue
        dt_idx = np.array(e["dtIds"], dtype=int) - 1
        matched = (e["dtMatches"][iou_idx, :] > 0) & (e["dtIgnore"][iou_idx, :] == 0)
        tp_valid[dt_idx[matched]] = True

    tp = np.zeros(len(dataset_in), dtype=bool)
    tp[np.where(valid)[0]] = tp_valid
    return tp


def resorted_raw_baseline(dataset_in, indices_r, run_tag, only_objects, data_space):
    n_data = len(dataset_in)
    raw_score = dataset_in["gdino_score"].to_numpy()
    is_obj = dataset_in["resort_true_tp"].to_numpy()

    in_budget = np.zeros(n_data, dtype=bool)
    in_budget[np.unique(indices_r)] = True

    resorted_score = raw_score.copy()
    budget_tp = in_budget & (is_obj == 1)
    budget_fp = in_budget & (is_obj == 0)
    resorted_score[budget_tp] = 1.0 - TP_TIE_BREAK_EPS * (1.0 - raw_score[budget_tp])
    resorted_score[budget_fp] = 0.0

    raw_probs = np.vstack(dataset_in["gdino_probs"].to_numpy())
    p_bg = np.prod(1.0 - raw_probs, axis=1)
    p_hat = np.hstack([raw_probs, p_bg[:, None]])
    p_hat = p_hat / p_hat.sum(axis=1, keepdims=True)

    soft = np.vstack(dataset_in["soft_label"].to_numpy())
    n_classes = soft.shape[1]

    n_tp = int(budget_tp.sum())
    u = np.random.uniform(size=(n_tp, 1))
    cdf = np.cumsum(soft[budget_tp], axis=1)
    sampled_class = np.clip((u > cdf).sum(axis=1), 0, n_classes - 1)
    one_hot = np.zeros((n_tp, n_classes))
    one_hot[np.arange(n_tp), sampled_class] = 1.0
    p_hat[budget_tp] = one_hot
    p_hat[budget_fp] = soft[budget_fp]

    dataset_out = dataset_in.copy()
    dataset_out["p_hat"] = list(p_hat)
    dataset_out["prediction"] = p_hat.argmax(axis=1)
    dataset_out["score_est"] = list(np.stack([1.0 - resorted_score, resorted_score], axis=1))
    _apply_object_conditioning(dataset_out)

    dataset_out.attrs["run_meta"] = dict(
        dataset_name=run_tag, only_objects=only_objects, data_space=data_space,
        label_column="label", prob_label_column="soft_label",
        n_data=n_data, k=None, alpha=None, n_samples=int(in_budget.sum()), config=None,
        elapsed_time=0.0, time_NN=0.0, time_graph=0.0, time_setup=0.0, time_solver=0.0, time_spread=0.0,
    )
    return dataset_out


def calibrated_gdino_baseline(dataset_in, df_full_in, budget, run_tag, only_objects, data_space, seed=42):
    np.random.seed(seed)
    n_sampled_full = max(1, int(len(df_full_in) * budget))
    indices_full = np.random.choice(len(df_full_in), size=n_sampled_full, replace=True)
    raw_score_full = df_full_in["gdino_score"].values
    match_full = df_full_in["gdino_match"].astype(float).values

    lr = LogisticRegression()
    lr.fit(raw_score_full[indices_full].reshape(-1, 1), match_full[indices_full])

    raw_probs = np.vstack(dataset_in["gdino_probs"].to_numpy())
    calibrated_probs = lr.predict_proba(raw_probs.ravel().reshape(-1, 1))[:, 1].reshape(raw_probs.shape)
    p_bg = np.prod(1.0 - calibrated_probs, axis=1)
    p_hat_baseline = np.hstack([calibrated_probs, p_bg[:, None]])
    p_hat_baseline = p_hat_baseline / p_hat_baseline.sum(axis=1, keepdims=True)

    dataset_out = dataset_in.copy()
    dataset_out["p_hat"] = list(p_hat_baseline)
    dataset_out["prediction"] = p_hat_baseline.argmax(axis=1)
    p_obj = 1.0 - p_hat_baseline[:, -1]
    dataset_out["score_est"] = list(np.stack([p_hat_baseline[:, -1], p_obj], axis=1))
    _apply_object_conditioning(dataset_out)

    dataset_out.attrs["run_meta"] = dict(
        dataset_name=run_tag, only_objects=only_objects, data_space=data_space,
        label_column="label", prob_label_column="soft_label",
        n_data=len(dataset_out), k=None, alpha=None, n_samples=n_sampled_full, config=None,
        elapsed_time=0.0, time_NN=0.0, time_graph=0.0, time_setup=0.0, time_solver=0.0, time_spread=0.0,
    )
    return dataset_out, p_hat_baseline


def build_bayes_dataset(kappa, p_hat_baseline, F_pls, template_ds, run_tag, pls_k, pls_alpha, n_samples_b,
                         only_objects, data_space, amgx_config, pls_bandwidth_scale=1.0):
    p_hat_unnorm = kappa * p_hat_baseline + F_pls
    p_hat = p_hat_unnorm / p_hat_unnorm.sum(axis=1, keepdims=True)

    ds = template_ds.copy()
    ds["p_hat"] = list(p_hat)
    ds["prediction"] = p_hat.argmax(axis=1)
    p_obj = 1.0 - p_hat[:, -1]
    ds["score_est"] = list(np.stack([p_hat[:, -1], p_obj], axis=1))
    _apply_object_conditioning(ds)

    ds.attrs["run_meta"] = dict(
        dataset_name=run_tag, only_objects=only_objects, data_space=data_space,
        label_column="label", prob_label_column="soft_label",
        n_data=len(ds), k=pls_k, alpha=pls_alpha, n_samples=n_samples_b, config=amgx_config,
        # the PLS fit's own bandwidth -- without it evaluate_result defaults to 1.0 and the
        # grid CSV's Bayes rows would misreport the graph they were built on
        bandwidth_scale=pls_bandwidth_scale,
        elapsed_time=0.0, time_NN=0.0, time_graph=0.0, time_setup=0.0, time_solver=0.0, time_spread=0.0,
    )
    return ds


def sample_budget_indices(n_total, budget, seed=42):
    np.random.seed(seed)
    n = max(1, int(n_total * budget))
    return np.random.choice(n_total, size=n, replace=True)


def run_grid_batched(method_name, hp_list, fit_fn, evaluate_fn, n_jobs, track_best_by=None):
    rows = []
    best = {"value": np.inf, "ds": None, "hp": None}
    for start in range(0, len(hp_list), n_jobs):
        batch = hp_list[start:start + n_jobs]
        fitted = [fit_fn(hp) for hp in batch]
        results = Parallel(n_jobs=n_jobs)(delayed(evaluate_fn)(ds) for ds in fitted)
        for hp, ds, res in zip(batch, fitted, results):
            row = res.iloc[0].to_dict()
            row.update(method=method_name, **hp)
            rows.append(row)
            if track_best_by is not None and row[track_best_by] < best["value"]:
                best.update(value=row[track_best_by], ds=ds, hp=hp)
        done = min(start + n_jobs, len(hp_list))
        print(f"    {method_name}: {done}/{len(hp_list)} combos done", flush=True)
    return (rows, best) if track_best_by is not None else rows


def main(args):
    dataset_path = args.dataset_path
    soft_gt_path = args.soft_gt_path
    amgx_config = args.amgx_config
    results_dir = args.results_dir
    run_name = args.run_name
    score_thr = args.score_thr
    only_objects = False
    data_space = "gdino_embedding"
    compute_ap, compute_as = True, True
    soft_label_thresh = 0.5
    n_jobs = args.n_jobs

    if args.quick:
        pls_k_grid, pls_alpha_grid, pls_bandwidth_grid = PLS_K_GRID_QUICK, PLS_ALPHA_GRID_QUICK, PLS_BANDWIDTH_GRID_QUICK
        gkr_gamma_grid, knn_k_grid = GKR_GAMMA_GRID_QUICK, KNN_K_GRID_QUICK
        bayes_kappa_grid, budgets = BAYES_KAPPA_GRID_QUICK, BUDGETS_QUICK
    else:
        pls_k_grid, pls_alpha_grid, pls_bandwidth_grid = PLS_K_GRID_FULL, PLS_ALPHA_GRID_FULL, PLS_BANDWIDTH_GRID_FULL
        gkr_gamma_grid, knn_k_grid = GKR_GAMMA_GRID_FULL, KNN_K_GRID_FULL
        bayes_kappa_grid, budgets = BAYES_KAPPA_GRID_FULL, BUDGETS_FULL

    gkr_m = max(500, max(knn_k_grid))
    os.makedirs(results_dir, exist_ok=True)
    print(f"N_JOBS = {n_jobs}", flush=True)

    def _evaluate(ds):
        return evaluate_result(ds, compute_ap=compute_ap, compute_as=compute_as,
                                soft_gt_path=soft_gt_path, soft_label_thresh=soft_label_thresh)

    # ── 1  Load & prepare dataset ──────────────────────────────────────────────
    df_full = pd.read_parquet(dataset_path)
    print(f"Full dataset : {len(df_full):,} detections  |  "
          f"{df_full['gdino_match'].sum():,} matched to soft GT  "
          f"({df_full['gdino_match'].mean():.1%})", flush=True)

    dataset = df_full[df_full["gdino_score"] >= score_thr].copy().reset_index(drop=True)
    print(f"After score >= {score_thr}: {len(dataset):,} detections  |  "
          f"{dataset['gdino_match'].sum():,} matched  "
          f"({dataset['gdino_match'].mean():.1%})", flush=True)

    sl = np.array([v for v in dataset["soft_label"].values])
    dataset["soft_label"] = list(sl)
    dataset["is_object"] = dataset["gdino_match"].astype(float)

    dataset["resort_true_tp"] = _category_aware_tp(dataset, soft_gt_path, soft_label_thresh)
    _gd_match_bool = dataset["gdino_match"].astype(bool)
    print(f"Resort baseline ground truth -- gdino_match TP: {int(_gd_match_bool.sum()):,}  |  "
          f"category-aware TP: {int(dataset['resort_true_tp'].sum()):,}  |  "
          f"rows differing: {int((_gd_match_bool != dataset['resort_true_tp']).sum()):,} "
          f"({(_gd_match_bool != dataset['resort_true_tp']).mean():.2%})", flush=True)

    image_ids_unique = dataset["image_id"].unique()
    print(f"\n{len(dataset):,} detections across {len(image_ids_unique):,} images  "
          f"(dim {sl.shape[1]} soft labels)", flush=True)

    print(f"PLS:   {len(pls_k_grid)} k x {len(pls_bandwidth_grid)} bandwidth x {len(pls_alpha_grid)} alpha = "
          f"{len(pls_k_grid) * len(pls_bandwidth_grid) * len(pls_alpha_grid)} combos", flush=True)
    print(f"GKR:   {len(gkr_gamma_grid)} gamma values", flush=True)
    print(f"kNN:   {len(knn_k_grid)} k values", flush=True)
    print(f"Bayes: {len(bayes_kappa_grid)} kappa values", flush=True)
    print(f"Budgets: {budgets}", flush=True)

    # ── 2  Grid Search ──────────────────────────────────────────────────────────
    # Resume support: a checkpoint CSV is written after every *complete* budget
    # (never mid-budget -- see the checkpoint write at the end of the loop body),
    # so any budget already present in an existing results CSV can be trusted as
    # done and skipped, picking up where an interrupted run left off instead of
    # re-paying hours of already-completed GPU/CPU work.
    grid_csv_path = os.path.join(results_dir, f"{run_name}_grid_search_results.csv")
    grid_rows = []
    done_budgets = set()
    if os.path.exists(grid_csv_path):
        prior = pd.read_csv(grid_csv_path)
        grid_rows = prior.to_dict("records")
        done_budgets = set(prior["budget"].unique())
        print(f"Resuming from checkpoint: {len(prior)} rows already done for budgets "
              f"{sorted(done_budgets)}", flush=True)

    for budget in budgets:
        if budget in done_budgets:
            print(f"\n===== budget = {budget:.0%} ===== (skipping, already in checkpoint)", flush=True)
            continue
        t0 = time.time()
        print(f"\n===== budget = {budget:.0%} =====", flush=True)

        indices_b = sample_budget_indices(len(dataset), budget)
        n_samples_b = len(indices_b)
        neighbor_cache_b = _faiss_labeled_neighbors(dataset, data_space, indices_b, gkr_m)

        # PLS
        pls_hp = [{"k": k, "alpha": a, "bandwidth_scale": b}
                  for k in pls_k_grid for b in pls_bandwidth_grid for a in pls_alpha_grid]
        pls_rows, pls_best = run_grid_batched(
            "PLS", pls_hp,
            fit_fn=lambda hp: label_spreading_object_conventional(
                f"{run_name}_grid_PLS_b{budget}_k{hp['k']}_bw{hp['bandwidth_scale']:.3f}_a{hp['alpha']:.4f}",
                only_objects, dataset.copy(),
                data_space, "label", "soft_label", "is_object", "all", hp["k"], hp["alpha"], n_samples_b,
                indices=indices_b, config=amgx_config, bandwidth_scale=hp["bandwidth_scale"],
            ),
            evaluate_fn=_evaluate, n_jobs=n_jobs, track_best_by="RMSE",
        )
        for row in pls_rows:
            row["budget"], row["n_samples"] = budget, n_samples_b
        grid_rows.extend(pls_rows)
        print(f"  PLS best (lowest RMSE): k={pls_best['hp']['k']}  bandwidth_scale={pls_best['hp']['bandwidth_scale']:.3f}  "
              f"alpha={pls_best['hp']['alpha']:.4f}  RMSE={pls_best['value']:.5f}", flush=True)

        # GKR
        gkr_hp = [{"gamma": g} for g in gkr_gamma_grid]
        gkr_rows = run_grid_batched(
            "GKR", gkr_hp,
            fit_fn=lambda hp: gkr_object(
                f"{run_name}_grid_GKR_b{budget}_g{hp['gamma']:.4g}", only_objects, dataset.copy(), data_space,
                "label", "soft_label", "is_object", "all", hp["gamma"], n_samples_b,
                indices=indices_b, m=gkr_m, neighbor_cache=neighbor_cache_b,
            ),
            evaluate_fn=_evaluate, n_jobs=n_jobs,
        )
        for row in gkr_rows:
            row["budget"], row["n_samples"] = budget, n_samples_b
        grid_rows.extend(gkr_rows)

        # kNN
        knn_hp = [{"k": k} for k in knn_k_grid]
        knn_rows = run_grid_batched(
            "kNN", knn_hp,
            fit_fn=lambda hp: knn_regression_object(
                f"{run_name}_grid_kNN_b{budget}_k{hp['k']}", only_objects, dataset.copy(), data_space,
                "label", "soft_label", "is_object", "all", hp["k"], n_samples_b,
                indices=indices_b, neighbor_cache=neighbor_cache_b,
            ),
            evaluate_fn=_evaluate, n_jobs=n_jobs,
        )
        for row in knn_rows:
            row["budget"], row["n_samples"] = budget, n_samples_b
        grid_rows.extend(knn_rows)

        # Baseline (calibrated GDino, no hyperparameter)
        dataset_baseline_b, p_hat_baseline_b = calibrated_gdino_baseline(
            dataset, df_full, budget, f"{run_name}_grid_Baseline_b{budget}", only_objects, data_space,
        )
        row = _evaluate(dataset_baseline_b).iloc[0].to_dict()
        row.update(method="Baseline", budget=budget, n_samples=n_samples_b)
        grid_rows.append(row)

        # Bayes: blend PLS's own best-RMSE fit with the calibrated-GDino prior
        pls_trials_best = pls_best["ds"]["trials"].to_numpy()
        pls_p_hat_best = np.vstack(pls_best["ds"]["p_hat"].to_numpy())
        F_pls_best = pls_p_hat_best * pls_trials_best[:, None]

        bayes_hp = [{"kappa": kp} for kp in bayes_kappa_grid]
        bayes_rows = run_grid_batched(
            "Bayes", bayes_hp,
            fit_fn=lambda hp: build_bayes_dataset(
                hp["kappa"], p_hat_baseline_b, F_pls_best, pls_best["ds"],
                f"{run_name}_grid_Bayes_b{budget}_kappa{hp['kappa']:.4g}",
                pls_best["hp"]["k"], pls_best["hp"]["alpha"], n_samples_b,
                only_objects, data_space, amgx_config,
                pls_bandwidth_scale=pls_best["hp"]["bandwidth_scale"],
            ),
            evaluate_fn=_evaluate, n_jobs=n_jobs,
        )
        for row in bayes_rows:
            row["budget"], row["n_samples"] = budget, n_samples_b
        grid_rows.extend(bayes_rows)

        # Resort (no hyperparameter)
        dataset_resort_b = resorted_raw_baseline(dataset, indices_b, f"{run_name}_grid_Resort_b{budget}",
                                                  only_objects, data_space)
        row = _evaluate(dataset_resort_b).iloc[0].to_dict()
        row.update(method="Resort", budget=budget, n_samples=n_samples_b)
        grid_rows.append(row)

        n_runs = len(pls_hp) + len(gkr_hp) + len(knn_hp) + len(bayes_hp) + 2
        print(f"  budget {budget:.0%} done in {time.time() - t0:.0f}s  ({n_runs} runs)", flush=True)

        # Checkpoint after every budget -- a 6h run shouldn't lose everything to a
        # crash/OOM on the last budget.
        pd.DataFrame(grid_rows).to_csv(
            os.path.join(results_dir, f"{run_name}_grid_search_results.csv"), index=False)

    grid_results = pd.DataFrame(grid_rows)
    print(f"\nTotal grid search rows: {len(grid_results):,}", flush=True)

    grid_csv = os.path.join(results_dir, f"{run_name}_grid_search_results.csv")
    grid_results.to_csv(grid_csv, index=False)
    print(f"Saved grid search results -> {grid_csv}", flush=True)

    best_ap = (
        grid_results.sort_values("AP", ascending=False)
        .groupby(["method", "budget"], as_index=False).first()[["method", "budget", "AP"]]
        .rename(columns={"AP": "best_AP"})
    )
    best_as = (
        grid_results.sort_values("AS", ascending=False)
        .groupby(["method", "budget"], as_index=False).first()[["method", "budget", "AS"]]
        .rename(columns={"AS": "best_AS"})
    )
    grid_best = best_ap.merge(best_as, on=["method", "budget"])
    grid_best_csv = os.path.join(results_dir, f"{run_name}_grid_search_best.csv")
    grid_best.to_csv(grid_best_csv, index=False)
    print(f"Saved best-AP/best-AS-per-budget summary -> {grid_best_csv}", flush=True)
    print(grid_best, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True, help="matched_dataset parquet path")
    parser.add_argument("--soft_gt_path", required=True, help="soft-GT json path (categories/images/objects schema)")
    parser.add_argument("--amgx_config", default="./configs/FGMRES_AGGREGATION.json")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--score_thr", type=float, default=0.2)
    parser.add_argument("--n_jobs", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--quick", action="store_true", help="tiny grids, for an end-to-end smoke test")
    args = parser.parse_args()
    main(args)
