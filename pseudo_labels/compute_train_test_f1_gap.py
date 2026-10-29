"""Train-vs-test F1 gap diagnostic (circularity check) across all 15 (fold,
budget) PLS pseudo-label combos from generate_pls_pseudo_labels_sweep.py.

For each combo, reruns the FINAL PLS+Bayes step (winning k/alpha/kappa from that
combo's own honest search, full labeled fold as supervision -- exactly what the
pseudo-label generation itself does) to get p_bayes over every detection, then
reports F1 at the recorded threshold on two populations:
  - "train": the labeled fold's own detections (circular -- these ARE the
    supervision, so near-perfect self-consistency here doesn't mean anything)
  - "test": the genuinely unlabeled population (gdino_match used as a research-
    only diagnostic oracle, never touched by supervision)
A large gap between the two indicates that the self-consistency on the
supervised detections overstates the quality on unlabeled ones.

Reuses the same shared-graph-per-k caching as the pseudo-label sweep script.
Long job -- run via tmux, not interactively.
"""
import os
import sys
import json
import time

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve

repo_root = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
label_spreading_dir = os.path.join(repo_root, "label_spreading")
sys.path.insert(0, repo_root)
sys.path.insert(0, label_spreading_dir)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from src import spreading_conventional, set_graph

st_dir = "/home/klees/SoftTeacher/"
BUDGETS = [1, 5, 10]
FOLDS = [1, 2, 3, 4, 5]

PLS_DATASET_PATH = os.path.join(repo_root, "data/matched_dataset/coco_train_matched.parquet")
AMGX_CONFIG = os.path.join(label_spreading_dir, "configs/FGMRES_AGGREGATION.json")
PLS_SCORE_THR = 0.2

cal_dir = os.path.join(repo_root, "pseudo_labels/grounding_dino_pseudo_labels")
bayes_dir = os.path.join(repo_root, "pseudo_labels/bayesian_pls_pseudo_labels")
OUT_CSV = os.path.join(bayes_dir, "train_test_f1_gap_summary.csv")

# fold 1 @ 10% pseudo-labels were produced with the same honest search grid before the
# sweep script logged hyperparameters to pls_hp_search_*.json, so no JSON exists for that
# combo; its selected values are hardcoded here.
FOLD1_10PCT_HP = dict(k=50.0, alpha=0.7, kappa=0.5, threshold=0.3710, f1=0.7891)


def best_f1_at_threshold(y_true, s, thr):
    kept = s >= thr
    tp = int((kept & (y_true == 1)).sum())
    fp = int((kept & (y_true == 0)).sum())
    fn = int((~kept & (y_true == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return f1, p, r


def load_hp(fold, perc_labeled):
    if fold == 1 and perc_labeled == 10:
        return dict(FOLD1_10PCT_HP)
    path = os.path.join(bayes_dir, f"pls_hp_search_fold{fold}_{perc_labeled}pct.json")
    with open(path) as f:
        return json.load(f)


def main():
    print("Loading PLS matched dataset (COCO train)...", flush=True)
    pls_df_full = pd.read_parquet(PLS_DATASET_PATH)
    pls_dataset = pls_df_full[pls_df_full["gdino_score"] >= PLS_SCORE_THR].copy().reset_index(drop=True)
    pls_dataset["soft_label"] = list(np.array([v for v in pls_dataset["soft_label"].values]))
    pls_dataset["is_object"] = pls_dataset["gdino_match"].astype(float)
    classes = sorted(pls_dataset["label"].dropna().unique())
    n_data = len(pls_dataset)
    print(f"PLS dataset: {n_data:,} detections", flush=True)

    combos = [(fold, perc) for perc in BUDGETS for fold in FOLDS]
    hps = {(fold, perc): load_hp(fold, perc) for fold, perc in combos}
    unique_ks = sorted(set(hp["k"] for hp in hps.values()))
    print(f"Unique k values across all 15 combos' winning hyperparameters: {unique_ks}", flush=True)

    print("Building similarity graphs (shared across combos with the same k)...", flush=True)
    S_by_k = {}
    for k in unique_ks:
        t0 = time.time()
        S_by_k[int(k)], _, time_NN, time_graph = set_graph(int(k), pls_dataset, "gdino_embedding", n_data=n_data)
        print(f"  k={int(k)}: graph built in {time.time() - t0:.1f}s (NN={time_NN:.1f}s, graph={time_graph:.1f}s)", flush=True)

    rows = []
    for fold, perc_labeled in combos:
        t0 = time.time()
        hp = hps[(fold, perc_labeled)]
        k, alpha, kappa, threshold = int(hp["k"]), hp["alpha"], hp["kappa"], hp["threshold"]
        print(f"\n[{perc_labeled}% / fold {fold}]  k={k} alpha={alpha} kappa={kappa} threshold={threshold:.4f}", flush=True)

        labels_json = os.path.join(
            st_dir, f"data/coco/annotations/semi_supervised/instances_train2017.{fold}@{perc_labeled}.json"
        )
        with open(labels_json) as f:
            labels = json.load(f)
        labeled_image_ids = set(int(img["id"]) for img in labels["images"])

        pls_indices = np.where(pls_dataset["image_id"].isin(labeled_image_ids))[0]
        labeled_mask = pls_dataset["image_id"].isin(labeled_image_ids).to_numpy()
        unlabeled_mask = ~labeled_mask

        cal_path = os.path.join(cal_dir, f"calibration_inputs_fold{fold}_{perc_labeled}pct.json")
        with open(cal_path) as f:
            cal_payload = json.load(f)
        scores_full = np.array(cal_payload["scores"])
        TP_mask_full = np.array(cal_payload["TP_mask"])
        ir_full = IsotonicRegression(out_of_bounds="clip")
        ir_full.fit(scores_full.reshape(-1, 1), TP_mask_full)

        S = S_by_k[k]
        F, F_object, _, _, _ = spreading_conventional(
            alpha, S, pls_dataset, "soft_label", "is_object", classes, AMGX_CONFIG, pls_indices,
        )
        trials = F.sum(axis=1)
        obj_row_sum = F_object.sum(axis=1)
        p_pls = np.divide(F_object[:, 1], obj_row_sum, out=np.full(n_data, 0.5), where=obj_row_sum > 0)

        prior_obj = ir_full.predict(pls_dataset["gdino_score"].to_numpy())
        F_obj_1 = p_pls * trials
        F_obj_0 = (1.0 - p_pls) * trials
        unnorm_1 = kappa * prior_obj + F_obj_1
        unnorm_0 = kappa * (1 - prior_obj) + F_obj_0
        p_bayes = unnorm_1 / (unnorm_0 + unnorm_1)

        y_true = pls_dataset["gdino_match"].to_numpy()

        train_f1, train_p, train_r = best_f1_at_threshold(y_true[labeled_mask], p_bayes[labeled_mask], threshold)
        test_f1, test_p, test_r = best_f1_at_threshold(y_true[unlabeled_mask], p_bayes[unlabeled_mask], threshold)

        row = dict(fold=fold, budget=perc_labeled, k=k, alpha=alpha, kappa=kappa, threshold=threshold,
                   search_val_f1=hp["f1"], train_f1=train_f1, train_precision=train_p, train_recall=train_r,
                   test_f1=test_f1, test_precision=test_p, test_recall=test_r, gap=train_f1 - test_f1)
        rows.append(row)
        print(f"  train F1={train_f1:.4f} (P={train_p:.4f} R={train_r:.4f})   "
              f"test F1={test_f1:.4f} (P={test_p:.4f} R={test_r:.4f})   gap={train_f1 - test_f1:+.4f}   "
              f"[{time.time() - t0:.0f}s]", flush=True)

    result_df = pd.DataFrame(rows)
    result_df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved -> {OUT_CSV}", flush=True)
    print("\n=== Summary by budget ===", flush=True)
    print(result_df.groupby("budget")[["train_f1", "test_f1", "gap"]].agg(["mean", "std"]).round(4), flush=True)


if __name__ == "__main__":
    main()
