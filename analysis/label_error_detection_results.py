"""Paper-ready results display for `label_error_detection/parameter_sweep.py`'s
checkpoint (reference-consistent label-error definition, all 4 datasets,
score threshold 0.01). Run AFTER that sweep completes.

Produces:
  1. A 3-row (Random / GDino / Bayesian PLS) x (4 datasets x {AP, F1})
     comparison table per threshold, resized to \\linewidth, bold best /
     underlined second-best per (dataset, metric) column.
  2. A winning-hyperparameters table (k, alpha, bandwidth, kappa per dataset)
     per threshold.
  3. One precision-recall curve plot per dataset per threshold (appendix).

Reads everything directly from `parameter_sweep.py`'s checkpoint pickle --
precision/recall/AP/hyperparameters for every method are already there from
that script's own final runs, so nothing here re-solves PLS or touches the
matched datasets/label-error files at all.

No pure-PLS (kappa=0) row is reported: it was far below both GDino and the
Bayesian blend on held-out AP on every dataset.
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../utils")
from plot_layout import set_plot_layout

RESULTS_PICKLE = "../label_error_detection/logs/parameter_sweep_all_results.pkl"
REPORTED_SCORE_THRESHOLDS = [0.01]   # parameter_sweep.py's checkpoint also has thr=0.1, which is not reported

DATASET_ORDER = ["coco_val", "cityscapes_trainval", "kitti_train", "voc_trainval"]
DATASET_DISPLAY_NAMES = {"coco_val": "COCO", "cityscapes_trainval": "Cityscapes",
                          "kitti_train": "KITTI", "voc_trainval": "Pascal VOC"}
METHOD_ORDER = ["random", "gdino", "unlabeled_fp"]
METHOD_LABELS = {"random": "Random", "gdino": "GDino", "unlabeled_fp": "Bayesian PLS"}


def load_all_results(pickle_path=RESULTS_PICKLE):
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


# ── LaTeX table -- datasets as column groups (AP, F1), bold best / underline 2nd-best ──
def build_comparison_table(results_by_dataset, score_thr, out_path="tables/label_error_detection_table.tex"):
    """results_by_dataset: {dataset_key: {method: {..., 'ap':, 'f1':}}}"""
    rows = {method: {} for method in METHOD_ORDER}
    for dataset_key in DATASET_ORDER:
        for method in METHOD_ORDER:
            res = results_by_dataset[dataset_key][method]
            rows[method][(dataset_key, "AP")] = res["ap"] * 100
            rows[method][(dataset_key, "F1")] = res["f1"] * 100

    # rank per (dataset, metric) column across the 3 methods -> bold best, underline 2nd-best
    formatted = {method: {} for method in METHOD_ORDER}
    for dataset_key in DATASET_ORDER:
        for metric in ["AP", "F1"]:
            col = [(method, rows[method][(dataset_key, metric)]) for method in METHOD_ORDER]
            ranked = sorted(col, key=lambda t: t[1], reverse=True)
            best_method, second_method = ranked[0][0], ranked[1][0]
            for method, val in col:
                s = f"{val:.2f}"
                if method == best_method:
                    s = rf"\textbf{{{s}}}"
                elif method == second_method:
                    s = rf"\underline{{{s}}}"
                formatted[method][(dataset_key, metric)] = s

    header_names = " & ".join(rf"\multicolumn{{2}}{{c}}{{{DATASET_DISPLAY_NAMES[dk]}}}" for dk in DATASET_ORDER)
    cmidrules = " ".join(rf"\cmidrule(lr){{{2*i+2}-{2*i+3}}}" for i in range(len(DATASET_ORDER)))
    subheader = " & ".join("AP & F1" for _ in DATASET_ORDER)
    col_spec = "l" + "rr" * len(DATASET_ORDER)

    body_rows = []
    for method in METHOD_ORDER:
        cells = " & ".join(formatted[method][(dk, metric)] for dk in DATASET_ORDER for metric in ["AP", "F1"])
        body_rows.append(f"{METHOD_LABELS[method]} & {cells} \\\\")

    table = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Label-error detection: AP and best F1 (both \%) for ranking originally-false-positive "
        rf"detections against the reference-consistent label-error set (IoU $\geq 0.1$), at SCORE\_THR={score_thr}. "
        r"Bold = best, underline = second-best per column. Bayesian PLS blends the PLS-propagated "
        r"objectness score with calibrated GDino score at the AP-selected kappa.}",
        rf"\label{{tab:label_error_detection_thr{str(score_thr).replace('.', '')}}}",
        r"\resizebox{\linewidth}{!}{%",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        f" & {header_names} \\\\",
        cmidrules,
        f"Method & {subheader} \\\\",
        r"\midrule",
        *body_rows,
        r"\bottomrule",
        r"\end{tabular}%",
        r"}",
        r"\end{table}",
    ])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(table + "\n")
    return table


# ── winning-hyperparameter table -- lets a reader check e.g. whether a
# near-tied dataset's blend just collapsed onto the prior (kappa -> large) rather
# than genuinely using PLS's propagated evidence ──────────────────────────────
def build_hyperparam_table(results_by_dataset, score_thr, out_path="tables/label_error_detection_hyperparams.tex"):
    rows = []
    for dataset_key in DATASET_ORDER:
        hb = results_by_dataset[dataset_key]["unlabeled_fp"]
        kappa_str = r"$\infty$" if np.isinf(hb["kappa"]) else f"{hb['kappa']:g}"
        rows.append(
            rf"{DATASET_DISPLAY_NAMES[dataset_key]} & {int(hb['k'])} & {hb['alpha']:g} & "
            rf"{hb['bandwidth_scale']:g} & {kappa_str} \\"
        )

    table = "\n".join([
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Winning Bayesian-blend hyperparameters at SCORE\_THR={score_thr}, honest-search-selected "
        r"by AP on held-out original-GT TP/FP (never the label-error set itself). $\kappa$ close to 0 means "
        r"the blend relies almost entirely on PLS's own propagated evidence; $\kappa \to \infty$ would mean "
        r"it collapses onto the calibrated-GDino prior alone.}",
        rf"\label{{tab:label_error_detection_hyperparams_thr{str(score_thr).replace('.', '')}}}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dataset & $k$ & $\alpha$ & bandwidth & $\kappa$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        f.write(table + "\n")
    return table


# ── PR curves, one plot per dataset (appendix) ──────────────────────────
def plot_pr_curves_per_dataset(results_by_dataset, score_thr, out_dir="plots"):
    set_plot_layout(path_to_latex="/home/klees/texlive/2024/bin/x86_64-linux")
    os.makedirs(out_dir, exist_ok=True)
    COLORS = {"random": "gray", "gdino": "darkorange", "unlabeled_fp": "steelblue"}
    LINESTYLES = {"random": "--", "gdino": "-", "unlabeled_fp": "-"}
    saved = []
    for dataset_key in DATASET_ORDER:
        fig, ax = plt.subplots()
        for method in METHOD_ORDER:
            res = results_by_dataset[dataset_key][method]
            ax.plot(res["recall"], res["precision"], color=COLORS[method], ls=LINESTYLES[method],
                     linewidth=1.4, label=f"{METHOD_LABELS[method]} (AP={res['ap']*100:.1f})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, ls=":", alpha=0.6)
        ax.legend(fontsize=7.5, frameon=False)
        plt.tight_layout()
        out_path = f"{out_dir}/label_error_pr_curve_{dataset_key}_thr{score_thr}.pdf"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)
    return saved


if __name__ == "__main__":
    all_results = load_all_results()
    score_thresholds = [thr for thr in REPORTED_SCORE_THRESHOLDS if thr in all_results]
    print(f"Loaded {RESULTS_PICKLE}: checkpoint has SCORE_THR={sorted(all_results.keys())}, "
          f"reporting {score_thresholds}, "
          f"datasets={list(next(iter(all_results.values())).keys())}")

    for score_thr in score_thresholds:
        print(f"\n=== SCORE_THR={score_thr} ===")
        results_by_dataset = all_results[score_thr]
        for dataset_key in DATASET_ORDER:
            r = results_by_dataset[dataset_key]
            print(f"  {DATASET_DISPLAY_NAMES[dataset_key]}: random AP={r['random']['ap']:.4f}  "
                  f"gdino AP={r['gdino']['ap']:.4f}  bayes_pls AP={r['unlabeled_fp']['ap']:.4f}  "
                  f"(k={r['unlabeled_fp']['k']} alpha={r['unlabeled_fp']['alpha']} "
                  f"bandwidth={r['unlabeled_fp']['bandwidth_scale']} kappa={r['unlabeled_fp']['kappa']})")

        table = build_comparison_table(results_by_dataset, score_thr,
                                        out_path=f"tables/label_error_detection_table_thr{score_thr}.tex")
        print(table)

        hp_table = build_hyperparam_table(results_by_dataset, score_thr,
                                           out_path=f"tables/label_error_detection_hyperparams_thr{score_thr}.tex")
        print(hp_table)

        saved_plots = plot_pr_curves_per_dataset(results_by_dataset, score_thr)
        print(f"Saved PR-curve plots: {saved_plots}")

        np.savez(f"tables/label_error_detection_curves_thr{score_thr}.npz",
                 **{f"{dk}_{m}_{k}": results_by_dataset[dk][m][k]
                    for dk in DATASET_ORDER for m in METHOD_ORDER for k in ["precision", "recall"]})
        print(f"Saved curves -> tables/label_error_detection_curves_thr{score_thr}.npz")
