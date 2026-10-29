"""Generates Bayesian-PLS pseudo-labels for the "100% budget" SSOD setting:
ALL of COCO train2017 used as full supervision (both the labels themselves and
the parameter-tuning signal, via an internal 80/20 image-level split -- same
honest-search methodology as generate_pls_pseudo_labels_sweep.py's fold-based
combos), with the COCO unlabeled2017 split's own Grounding DINO detections
added into the PLS similarity graph as additional (unlabeled) nodes to be
pseudo-labeled by propagation from the fully-supervised train2017 side.

Mirrors grounding_dino_pseudo_labels.ipynb's own "100% budget" section (which
produced pseudo_labels_full_100pct.json via a much simpler calibrated-
threshold recipe, no graph propagation at all) but for the PLS+Bayes method,
following generate_pls_pseudo_labels_sweep.py's fold-based output convention:
pseudo_labels_full_100pct_pls_bayes.json in the same bayesian_pls_pseudo_labels/
directory as the 15 fold/budget combos.

Since coco_unlabeled2017 has no ground truth at all, every one of its
detections is -- by construction -- an "unmatched" node exactly like an
unmatched detection in coco_train_matched.parquet: soft_label = the 81-dim
zero vector, label = 80 (background/cantsolve sentinel), gdino_match = 0
(same convention build_matching_dataset_coco_train.ipynb uses for any
detection that doesn't match a GT box). These placeholder values are NEVER
read as supervision -- coco_unlabeled2017 rows are never included in the
sampled indices passed to PLS -- they only serve to place these detections as
k-NN graph nodes (via gdino_embedding) that get pseudo-labeled by propagation.

NOTE on calibration caching: grounding_dino_pseudo_labels/calibration_inputs_
full_100pct.json (built by the GDino-only baseline) has a per-detection
scores/TP_mask array but a *per-image* (not per-detection) image_ids array --
not usable here, since the honest search needs a flattened, per-detection
image_ids array aligned with scores/TP_mask to build the sup/val split mask.
This script therefore computes and caches its own per-detection version
(bayesian_pls_pseudo_labels/calibration_inputs_full_100pct.json).

Long unattended job (large: ~2M train + several million unlabeled2017
detections after score filtering) -- run via tmux, not interactively.
Checkpointed: skips straight to printing a summary if the output pseudo-label
JSON already exists.
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

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

from calibration.calibrate import gather_detections_from_parquet, extract_per_image_scores, flatten_per_image_data
from src import spreading_conventional, set_graph

st_dir = "/home/klees/SoftTeacher/"

TRAIN_MATCHED_PATH = os.path.join(repo_root, "data/matched_dataset/coco_train_matched.parquet")
UNLABELED_PRED_PATH = os.path.join(repo_root, "inference/predictions/gdino/coco_unlabeled/predictions.parquet")
TRAIN_PRED_PATH = os.path.join(repo_root, "inference/predictions/gdino/coco_train/predictions.parquet")
FULL_TRAIN_ANN = os.path.join(st_dir, "data/coco/annotations/instances_train2017.json")
UNLABELED_INFO_ANN = os.path.join(st_dir, "data/coco/annotations/image_info_unlabeled2017.json")

AMGX_CONFIG = os.path.join(label_spreading_dir, "configs/FGMRES_AGGREGATION.json")
PLS_SCORE_THR = 0.2
N_CLASSES = 80
BG_LABEL = 80

bayes_out_dir = os.path.join(repo_root, "pseudo_labels/bayesian_pls_pseudo_labels")
os.makedirs(bayes_out_dir, exist_ok=True)

# Same grids as generate_pls_pseudo_labels_sweep.py's honest search.
K_GRID_SEARCH = [10, 20, 50]
ALPHA_GRID_SEARCH = [0.7, 0.85, 0.9, 0.95, 0.99]
KAPPA_GRID_SEARCH = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]
SEARCH_VAL_FRAC = 0.2
SEARCH_SEED = 123

OUT_PATH = os.path.join(bayes_out_dir, "pseudo_labels_full_100pct_pls_bayes.json")
HP_LOG_PATH = os.path.join(bayes_out_dir, "pls_hp_search_full_100pct.json")
CAL_CACHE_PATH = os.path.join(bayes_out_dir, "calibration_inputs_full_100pct.json")


def best_f1(y_true, s):
    prec, rec, thr = precision_recall_curve(y_true, s)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec + 1e-12), 0.0)
    idx = np.argmax(f1[:-1])   # last point (recall=0) has no corresponding threshold
    return f1[idx], prec[idx], rec[idx], thr[idx]


def load_unlabeled_as_graph_nodes(coco_id_to_name):
    """Builds coco_unlabeled2017's rows in the same schema as coco_train_matched.parquet,
    but every row is trivially "unmatched" (no GT exists for this split at all)."""
    print("Loading COCO-unlabeled predictions (score-filtered, column-pruned)...", flush=True)
    t0 = time.time()
    df = pd.read_parquet(
        UNLABELED_PRED_PATH,
        columns=["image_id", "box", "score", "label", "last_hidden_state"],
        filters=[("score", ">=", PLS_SCORE_THR)],
    )
    print(f"  {len(df):,} detections across {df['image_id'].nunique():,} images "
          f"(score >= {PLS_SCORE_THR}) loaded in {time.time() - t0:.0f}s", flush=True)

    zero_81 = np.zeros(N_CLASSES + 1, dtype=np.float32)
    n = len(df)
    out = pd.DataFrame({
        "image_id": df["image_id"].to_numpy(),
        "gdino_bbox": df["box"].tolist(),
        "gdino_score": df["score"].astype(np.float32).to_numpy(),
        "gdino_label": [coco_id_to_name.get(int(l), "unknown") for l in df["label"].to_numpy()],
        "gdino_embedding": df["last_hidden_state"].tolist(),
        "gdino_match": np.zeros(n, dtype=np.int64),
        "soft_label": [zero_81.copy() for _ in range(n)],
        "label": np.full(n, BG_LABEL, dtype=np.int64),
    })
    return out


def get_calibration_inputs_full(labeled_image_ids):
    if os.path.exists(CAL_CACHE_PATH):
        print(f"Reusing cached calibration inputs from {CAL_CACHE_PATH}", flush=True)
        with open(CAL_CACHE_PATH) as f:
            payload = json.load(f)
        return np.array(payload["scores"]), np.array(payload["TP_mask"]), np.array(payload["image_ids"])

    print("Computing calibration inputs over ALL train2017 detections (unfiltered, "
          "one-time expensive COCOeval pass)...", flush=True)
    t0 = time.time()
    all_detections = gather_detections_from_parquet(list(labeled_image_ids), TRAIN_PRED_PATH)
    cocoGt = COCO(FULL_TRAIN_ANN)
    cocoDt = cocoGt.loadRes(all_detections)
    cocoEval = COCOeval(cocoGt, cocoDt, "bbox")
    cocoEval.params.iouThrs = [0.5]
    cocoEval.params.areaRng = [cocoEval.params.areaRng[0]]
    cocoEval.params.maxDets = [300]
    cocoEval.evaluate()
    cocoEval.accumulate()
    per_image = extract_per_image_scores(cocoEval)
    scores, TP_mask, img_ids_flat = flatten_per_image_data(per_image)
    print(f"  done in {time.time() - t0:.0f}s -- {len(scores):,} detections", flush=True)

    payload = {"scores": scores.tolist(), "TP_mask": [int(x) for x in TP_mask],
               "image_ids": img_ids_flat.tolist()}
    with open(CAL_CACHE_PATH, "w") as f:
        json.dump(payload, f)
    return scores, TP_mask, img_ids_flat


def main():
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            existing = json.load(f)
        print(f"Already exists: {OUT_PATH}")
        print(f"  Total images      : {len(existing['images']):,}")
        print(f"  Total annotations : {len(existing['annotations']):,}")
        return

    t_start = time.time()

    print("Loading COCO train (fully labeled side)...", flush=True)
    train_df = pd.read_parquet(TRAIN_MATCHED_PATH)
    train_df = train_df[train_df["gdino_score"] >= PLS_SCORE_THR].copy()
    keep_cols = ["image_id", "gdino_bbox", "gdino_score", "gdino_label", "gdino_embedding",
                 "gdino_match", "soft_label", "label"]
    train_df = train_df[keep_cols].reset_index(drop=True)
    print(f"  {len(train_df):,} train detections across {train_df['image_id'].nunique():,} images", flush=True)

    cocoGt_full = COCO(FULL_TRAIN_ANN)
    coco_cats = cocoGt_full.dataset["categories"]
    coco_id_to_name = {c["id"]: c["name"] for c in coco_cats}
    labeled_image_ids = set(cocoGt_full.imgs.keys())
    print(f"  {len(labeled_image_ids):,} total train2017 images (full GT)", flush=True)

    unlabeled_df = load_unlabeled_as_graph_nodes(coco_id_to_name)

    pls_dataset = pd.concat([train_df, unlabeled_df], ignore_index=True)
    pls_dataset["soft_label"] = list(np.array([v for v in pls_dataset["soft_label"].values]))
    pls_dataset["is_object"] = pls_dataset["gdino_match"].astype(float)
    classes = sorted(pls_dataset["label"].dropna().unique())
    n_data = len(pls_dataset)
    print(f"Combined PLS dataset: {n_data:,} detections "
          f"({len(train_df):,} train + {len(unlabeled_df):,} unlabeled)", flush=True)

    pls_indices = np.where(pls_dataset["image_id"].isin(labeled_image_ids))[0]
    unlabeled_mask = ~pls_dataset["image_id"].isin(labeled_image_ids)
    print(f"  {len(pls_indices):,} labeled (train2017) detections as supervision, "
          f"{int(unlabeled_mask.sum()):,} unlabeled2017 detections to pseudo-label", flush=True)

    print("\nBuilding similarity graphs for every k in K_GRID_SEARCH...", flush=True)
    S_by_k = {}
    for k in K_GRID_SEARCH:
        t0 = time.time()
        S_by_k[k], _, time_NN, time_graph = set_graph(k, pls_dataset, "gdino_embedding", n_data=n_data)
        print(f"  k={k}: graph built in {time.time() - t0:.1f}s (NN={time_NN:.1f}s, graph={time_graph:.1f}s)",
              flush=True)

    # ── calibration on the FULL labeled (train2017) set ─────────────────────────
    scores_full, TP_mask_full, cal_img_ids_full = get_calibration_inputs_full(labeled_image_ids)
    scores_full = np.asarray(scores_full)
    TP_mask_full = np.asarray(TP_mask_full)
    cal_img_ids_full = np.asarray(cal_img_ids_full)

    ir_full = IsotonicRegression(out_of_bounds="clip")
    ir_full.fit(scores_full.reshape(-1, 1), TP_mask_full)

    # ── honest hyperparameter search: 80/20 split of ALL train2017 images ───────
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

    print(f"\nHonest search: {len(search_sup_image_ids):,} sup images / {len(search_val_image_ids):,} val images "
          f"-> {len(search_sup_indices):,} sup detections / {int(search_val_mask.sum()):,} val detections",
          flush=True)

    t_search = time.time()
    best = {"f1": -1.0}
    for k in K_GRID_SEARCH:
        S = S_by_k[k]
        for alpha in ALPHA_GRID_SEARCH:
            F_s, F_obj_s, _, _, _ = spreading_conventional(
                alpha, S, pls_dataset, "soft_label", "is_object", classes, AMGX_CONFIG, search_sup_indices,
            )
            row_sum_s = F_s.sum(axis=1)
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
        print(f"  k={k} search done ({time.time() - t_search:.0f}s elapsed)", flush=True)

    print(f"Honest search winner: k={best['k']} alpha={best['alpha']} kappa={best['kappa']}  "
          f"F1={best['f1']:.4f} (P={best['precision']:.4f} R={best['recall']:.4f})  "
          f"threshold={best['threshold']:.4f}", flush=True)

    # ── final run: winning (k, alpha), FULL train2017 as supervision ────────────
    S_final = S_by_k[best["k"]]
    t_final = time.time()
    F_final, F_obj_final, _, _, _ = spreading_conventional(
        best["alpha"], S_final, pls_dataset, "soft_label", "is_object", classes, AMGX_CONFIG, pls_indices,
    )
    trials_final = F_final.sum(axis=1)
    obj_row_sum_final = F_obj_final.sum(axis=1)
    p_pls_final = np.divide(F_obj_final[:, 1], obj_row_sum_final, out=np.full(n_data, 0.5),
                             where=obj_row_sum_final > 0)

    prior_obj_final = ir_full.predict(pls_dataset["gdino_score"].to_numpy())
    F_pls_obj_1 = p_pls_final * trials_final
    F_pls_obj_0 = (1.0 - p_pls_final) * trials_final
    unnorm_1 = best["kappa"] * prior_obj_final + F_pls_obj_1
    unnorm_0 = best["kappa"] * (1 - prior_obj_final) + F_pls_obj_0
    p_bayes = unnorm_1 / (unnorm_0 + unnorm_1)
    print(f"Final full-supervision run done in {time.time() - t_final:.0f}s", flush=True)

    # ── build + save pseudo-label COCO training set ──────────────────────────────
    unlabeled_mask_np = unlabeled_mask.to_numpy()
    p_bayes_unlabeled = p_bayes[unlabeled_mask_np]
    unlabeled_rows = pls_dataset.loc[unlabeled_mask_np].reset_index(drop=True)
    keep_mask = p_bayes_unlabeled >= best["threshold"]
    kept = unlabeled_rows[keep_mask]
    kept_scores = p_bayes_unlabeled[keep_mask]

    with open(UNLABELED_INFO_ANN) as f:
        _unlabeled_info = json.load(f)
    unlabeled_img_info = {img["id"]: img for img in _unlabeled_info["images"]}
    categories = coco_cats
    name_to_id = {c["name"]: c["id"] for c in categories}

    next_ann_id = max(cocoGt_full.anns.keys(), default=0) + 1
    pseudo_annotations = []
    pseudo_image_ids = set()
    for image_id, gdino_bbox, gdino_label, score in tqdm(
            zip(kept["image_id"].to_numpy(), kept["gdino_bbox"].to_numpy(),
                kept["gdino_label"].to_numpy(), kept_scores),
            total=len(kept), desc="Building pseudo-annotations [full/100%]"):
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

    pseudo_images = [unlabeled_img_info[iid] for iid in pseudo_image_ids if iid in unlabeled_img_info]

    labeled_images_full = list(cocoGt_full.imgs.values())
    labeled_annotations_full = list(cocoGt_full.anns.values())

    coco_out = {
        "images": labeled_images_full + pseudo_images,
        "annotations": labeled_annotations_full + pseudo_annotations,
        "categories": categories,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(coco_out, f)

    with open(HP_LOG_PATH, "w") as f:
        json.dump({kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                    for kk, vv in best.items()}, f)

    print(f"\nSaved: {OUT_PATH}")
    print(f"  Total images      : {len(coco_out['images']):,}  "
          f"({len(labeled_images_full):,} full-train + {len(pseudo_images):,} unlabeled)")
    print(f"  Total annotations : {len(coco_out['annotations']):,}  "
          f"({len(labeled_annotations_full):,} GT + {len(pseudo_annotations):,} pseudo-labels)")
    print(f"  Total time: {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
