"""Generates Bayesian-PLS pseudo-labels for every (fold, budget) combination in
SoftTeacher's semi-supervised COCO splits (5 folds x {1, 5, 10}% = 15 combos):
honest (labeled-fold-only) (k, alpha, kappa) search via an internal 80/20
image-level split, PLS + calibrated-GDino Bayes blend, F1-optimal threshold from
the held-out search split.

Each combo gets its own honest search, but the k-NN similarity graph does not
depend on which images are labeled (it is built over the full ~2M-detection COCO
train dataset), so graphs for every k in K_GRID_SEARCH are built once and reused
across all combos' searches and final runs.

Long unattended job -- run via tmux, not interactively. Checkpointed per
(fold, budget) combo: a combo whose pseudo-label JSON already exists is
skipped, as in grounding_dino_pseudo_labels.ipynb's sweep. The fold 1 @ 10%
output pre-existed this script, so it has no pls_hp_search_*.json log.
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve
from tqdm import tqdm

repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
label_spreading_dir = os.path.join(repo_root, "label_spreading")
sys.path.insert(0, repo_root)
sys.path.insert(0, label_spreading_dir)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from calibration.calibrate import gather_detections_from_parquet, extract_per_image_scores, flatten_per_image_data
from src import spreading_conventional, set_graph

st_dir = "/home/klees/SoftTeacher/"
BUDGETS = [1, 5, 10]
FOLDS = [1, 2, 3, 4, 5]

PLS_DATASET_PATH = os.path.join(repo_root, "data/matched_dataset/coco_train_matched.parquet")
AMGX_CONFIG = os.path.join(label_spreading_dir, "configs/FGMRES_AGGREGATION.json")
PLS_SCORE_THR = 0.2

MODEL_NAME = "gdino"
DATASET_NAME = "coco_train"
pred_parquet = os.path.join(repo_root, f"inference/predictions/{MODEL_NAME}/{DATASET_NAME}/predictions.parquet")

cal_out_dir = os.path.join(repo_root, "pseudo_labels/grounding_dino_pseudo_labels")
bayes_out_dir = os.path.join(repo_root, "pseudo_labels/bayesian_pls_pseudo_labels")
os.makedirs(cal_out_dir, exist_ok=True)
os.makedirs(bayes_out_dir, exist_ok=True)

# honest-search grids
K_GRID_SEARCH = [10, 20, 50]
ALPHA_GRID_SEARCH = [0.7, 0.85, 0.9, 0.95, 0.99]
KAPPA_GRID_SEARCH = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
SEARCH_VAL_FRAC = 0.2
SEARCH_SEED = 123


def best_f1(y_true, s):
    prec, rec, thr = precision_recall_curve(y_true, s)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-12), 0.0)
    idx = np.argmax(f1[:-1])   # last point (recall=0) has no corresponding threshold
    return f1[idx], prec[idx], rec[idx], thr[idx]


def get_calibration_inputs(fold, perc_labeled, image_ids, labels_json):
    """Cached scores/TP_mask/image_ids for a (fold, budget) combo's own labeled
    images, cached to calibration_inputs_fold{fold}_{pct}pct.json (same cache as
    grounding_dino_pseudo_labels.ipynb's sweep)."""
    save_path = os.path.join(cal_out_dir, f"calibration_inputs_fold{fold}_{perc_labeled}pct.json")
    if os.path.exists(save_path):
        with open(save_path) as f:
            payload = json.load(f)
        return np.array(payload["scores"]), np.array(payload["TP_mask"]), np.array(payload["image_ids"])

    all_detections = gather_detections_from_parquet(image_ids, pred_parquet)
    cocoGt = COCO(labels_json)
    cocoDt = cocoGt.loadRes(all_detections)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.params.iouThrs = [0.5]
    cocoEval.params.areaRng = [cocoEval.params.areaRng[0]]
    cocoEval.params.maxDets = [300]
    cocoEval.evaluate()
    cocoEval.accumulate()
    per_image = extract_per_image_scores(cocoEval)
    scores, TP_mask, img_ids_flat = flatten_per_image_data(per_image)

    payload = {
        "scores": list(scores) if not hasattr(scores, "tolist") else scores.tolist(),
        "TP_mask": [int(x) for x in TP_mask],
        "image_ids": list(img_ids_flat) if not hasattr(img_ids_flat, "tolist") else img_ids_flat.tolist(),
    }
    with open(save_path, "w") as f:
        json.dump(payload, f)
    return np.array(payload["scores"]), np.array(payload["TP_mask"]), np.array(payload["image_ids"])


def run_combo(fold, perc_labeled, pls_dataset, classes, n_data, S_by_k):
    out_path = os.path.join(bayes_out_dir, f"pseudo_labels_fold{fold}_{perc_labeled}pct_pls_bayes.json")
    if os.path.exists(out_path):
        print(f"[{perc_labeled}% / fold {fold}] Already exists, skipping: {out_path}", flush=True)
        return

    t_combo = time.time()
    print(f"\n{'=' * 70}\n[{perc_labeled}% / fold {fold}]\n{'=' * 70}", flush=True)

    labels_json = os.path.join(
        st_dir, f"data/coco/annotations/semi_supervised/instances_train2017.{fold}@{perc_labeled}.json"
    )
    with open(labels_json) as f:
        labels = json.load(f)
    labeled_image_ids = set(int(img["id"]) for img in labels["images"])

    pls_indices = np.where(pls_dataset["image_id"].isin(labeled_image_ids))[0]
    unlabeled_mask = ~pls_dataset["image_id"].isin(labeled_image_ids)
    print(f"  {len(labeled_image_ids):,} labeled images -> {len(pls_indices):,} labeled detections, "
          f"{int(unlabeled_mask.sum()):,} unlabeled detections", flush=True)

    # ── calibration on the FULL labeled fold (for the final Bayes prior) ──────
    cal_image_ids = [int(img["id"]) for img in labels["images"]]
    scores_full, TP_mask_full, cal_img_ids_full = get_calibration_inputs(fold, perc_labeled, cal_image_ids, labels_json)
    ir_full = IsotonicRegression(out_of_bounds="clip")
    ir_full.fit(scores_full.reshape(-1, 1), TP_mask_full)

    # ── honest hyperparameter search: 80/20 split within THIS combo's own labeled images ──
    labeled_image_ids_arr = np.array(sorted(labeled_image_ids))
    rng_search = np.random.default_rng(SEARCH_SEED)
    perm = rng_search.permutation(len(labeled_image_ids_arr))
    n_val = max(1, int(len(labeled_image_ids_arr) * SEARCH_VAL_FRAC))
    search_val_image_ids = set(labeled_image_ids_arr[perm[:n_val]].tolist())
    search_sup_image_ids = set(labeled_image_ids_arr[perm[n_val:]].tolist())

    sup_cal_mask = np.isin(cal_img_ids_full, list(search_sup_image_ids))
    ir_search = IsotonicRegression(out_of_bounds="clip")
    ir_search.fit(scores_full[sup_cal_mask].reshape(-1, 1), TP_mask_full[sup_cal_mask])

    search_sup_mask = pls_dataset["image_id"].isin(search_sup_image_ids).to_numpy()
    search_val_mask = pls_dataset["image_id"].isin(search_val_image_ids).to_numpy()
    search_sup_indices = np.where(search_sup_mask)[0]

    y_val = pls_dataset["gdino_match"].to_numpy()[search_val_mask]
    raw_score_val = pls_dataset["gdino_score"].to_numpy()[search_val_mask]
    prior_val = ir_search.predict(raw_score_val)

    best = {"f1": -1.0}
    for k in K_GRID_SEARCH:
        S = S_by_k[k]
        for alpha in ALPHA_GRID_SEARCH:
            F_s, F_obj_s, _, _, _ = spreading_conventional(
                alpha, S, pls_dataset, "soft_label", "is_object", classes, AMGX_CONFIG, search_sup_indices,
            )
            row_sum_s = F_s.sum(axis=1)                # "trials" -- class-posterior propagated mass
            obj_row_sum_s = F_obj_s.sum(axis=1)
            p_pls_s = np.divide(F_obj_s[:, 1], obj_row_sum_s, out=np.full(n_data, 0.5), where=obj_row_sum_s > 0)

            p_pls_val = p_pls_s[search_val_mask]
            trials_val = row_sum_s[search_val_mask]

            for kappa in KAPPA_GRID_SEARCH:
                unnorm_1 = kappa * prior_val + p_pls_val * trials_val
                unnorm_0 = kappa * (1 - prior_val) + (1 - p_pls_val) * trials_val
                p_bayes_val = unnorm_1 / (unnorm_0 + unnorm_1)
                f1, prec, rec, thr = best_f1(y_val, p_bayes_val)
                if f1 > best["f1"]:
                    best = dict(f1=f1, k=k, alpha=alpha, kappa=kappa, precision=prec, recall=rec, threshold=thr)
        print(f"  k={k} search done ({time.time() - t_combo:.0f}s elapsed)", flush=True)

    print(f"  Honest search winner: k={best['k']} alpha={best['alpha']} kappa={best['kappa']}  "
          f"F1={best['f1']:.4f} (P={best['precision']:.4f} R={best['recall']:.4f})  "
          f"threshold={best['threshold']:.4f}", flush=True)

    # ── final run: winning (k, alpha), FULL labeled set as supervision ────────
    S_final = S_by_k[best["k"]]
    F_final, F_obj_final, _, _, _ = spreading_conventional(
        best["alpha"], S_final, pls_dataset, "soft_label", "is_object", classes, AMGX_CONFIG, pls_indices,
    )
    trials_final = F_final.sum(axis=1)
    obj_row_sum_final = F_obj_final.sum(axis=1)
    p_pls_final = np.divide(F_obj_final[:, 1], obj_row_sum_final, out=np.full(n_data, 0.5), where=obj_row_sum_final > 0)

    prior_obj_final = ir_full.predict(pls_dataset["gdino_score"].to_numpy())
    F_pls_obj_1 = p_pls_final * trials_final
    F_pls_obj_0 = (1.0 - p_pls_final) * trials_final
    unnorm_1 = best["kappa"] * prior_obj_final + F_pls_obj_1
    unnorm_0 = best["kappa"] * (1 - prior_obj_final) + F_pls_obj_0
    p_bayes = unnorm_1 / (unnorm_0 + unnorm_1)

    # ── build + save pseudo-label COCO training set ────────────────────────────
    unlabeled_mask_np = unlabeled_mask.to_numpy()
    p_bayes_unlabeled = p_bayes[unlabeled_mask_np]
    unlabeled_rows = pls_dataset.loc[unlabeled_mask_np].reset_index(drop=True)
    keep_mask = p_bayes_unlabeled >= best["threshold"]
    kept = unlabeled_rows[keep_mask]
    kept_scores = p_bayes_unlabeled[keep_mask]

    full_train_ann = os.path.join(st_dir, "data/coco/annotations/instances_train2017.json")
    with open(full_train_ann) as f:
        _full = json.load(f)
    full_img_info = {img["id"]: img for img in _full["images"]}
    categories = _full["categories"]
    name_to_id = {c["name"]: c["id"] for c in categories}

    next_ann_id = max(a["id"] for a in labels["annotations"]) + 1
    pseudo_annotations = []
    pseudo_image_ids = set()
    for image_id, gdino_bbox, gdino_label, score in tqdm(
            zip(kept["image_id"].to_numpy(), kept["gdino_bbox"].to_numpy(),
                kept["gdino_label"].to_numpy(), kept_scores),
            total=len(kept), desc=f"Building pseudo-annotations [{perc_labeled}% / fold {fold}]"):
        cat_id = name_to_id.get(gdino_label)
        if cat_id is None:
            continue
        x1, y1, x2, y2 = gdino_bbox
        w, h = float(x2 - x1), float(y2 - y1)
        pseudo_annotations.append({
            "id": next_ann_id, "image_id": int(image_id), "category_id": int(cat_id),
            "bbox": [float(x1), float(y1), w, h], "area": w * h, "score": float(score),
            "iscrowd": 0, "segmentation": [],
        })
        next_ann_id += 1
        pseudo_image_ids.add(int(image_id))

    pseudo_images = [full_img_info[iid] for iid in pseudo_image_ids if iid in full_img_info]

    coco_out = {
        "images": labels["images"] + pseudo_images,
        "annotations": labels["annotations"] + pseudo_annotations,
        "categories": categories,
    }
    with open(out_path, "w") as f:
        json.dump(coco_out, f)

    hp_log_path = os.path.join(bayes_out_dir, f"pls_hp_search_fold{fold}_{perc_labeled}pct.json")
    with open(hp_log_path, "w") as f:
        json.dump({kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv) for kk, vv in best.items()}, f)

    print(f"[{perc_labeled}% / fold {fold}] Saved: {out_path}  "
          f"({len(coco_out['images']):,} images, {len(coco_out['annotations']):,} annotations, "
          f"{len(pseudo_annotations):,} pseudo-labels)  [{time.time() - t_combo:.0f}s]", flush=True)


def main():
    print("Loading PLS matched dataset (COCO train)...", flush=True)
    pls_df_full = pd.read_parquet(PLS_DATASET_PATH)
    pls_dataset = pls_df_full[pls_df_full["gdino_score"] >= PLS_SCORE_THR].copy().reset_index(drop=True)
    pls_dataset["soft_label"] = list(np.array([v for v in pls_dataset["soft_label"].values]))
    pls_dataset["is_object"] = pls_dataset["gdino_match"].astype(float)
    classes = sorted(pls_dataset["label"].dropna().unique())
    n_data = len(pls_dataset)
    print(f"PLS dataset: {n_data:,} detections across {pls_dataset['image_id'].nunique():,} images", flush=True)

    print("\nBuilding similarity graphs for every k in K_GRID_SEARCH (shared across all 15 combos)...", flush=True)
    S_by_k = {}
    for k in K_GRID_SEARCH:
        t0 = time.time()
        S_by_k[k], _, time_NN, time_graph = set_graph(k, pls_dataset, "gdino_embedding", n_data=n_data)
        print(f"  k={k}: graph built in {time.time() - t0:.1f}s (NN={time_NN:.1f}s, graph={time_graph:.1f}s)", flush=True)

    for perc_labeled in BUDGETS:
        for fold in FOLDS:
            run_combo(fold, perc_labeled, pls_dataset, classes, n_data, S_by_k)

    print("\nAll 15 (fold, budget) combinations done.", flush=True)


if __name__ == "__main__":
    main()
