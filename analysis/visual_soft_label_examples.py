"""Agreement / disagreement / uncertain-GT example figures, parametrized by dataset
(COCO val, VOC, Cityscapes, KITTI).

Same selection and plotting logic as visual_soft_label_examples.ipynb: for detections with a
matched soft-GT box, compares the soft ground truth, our estimate (PLS+Bayes at 100%
budget, using the grid search's best-AP hyperparameters) and GDino alone (calibrated
baseline, no propagation).

Usage (run from this directory):
    python visual_soft_label_examples.py --dataset voc_trainval \\
        --data_dir ../data --experiments_dir experiments --images_root /home/klees/data

Figures go to plots/examples/<dataset>/{agreement,disagreement,uncertain_gt}_<i>.png
(COCO val figures go to plots/examples/ directly).
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, "../utils"))
sys.path.append(os.path.join(HERE, "../label_spreading"))
from plot_layout import set_plot_layout
from src import label_spreading_object_conventional
from baseline_comparison import calibrated_gdino_baseline, build_bayes_dataset, sample_budget_indices

# per dataset: matched parquet, soft-GT json (both under --data_dir), grid-search run
# name (under --experiments_dir) and image directories (under --images_root; VOC's
# trainval images are split over two directories)
DATASETS = {
    "coco_val": dict(
        parquet="matched_dataset/coco_val_matched.parquet",
        soft_gt="soft_datasets/soft_COCO_2017_val.json",
        run_name="coco_val_baseline_comp_simple",
        image_dirs=["coco/val2017"],
    ),
    "voc_trainval": dict(
        parquet="matched_dataset/pascal_voc_trainval_matched.parquet",
        soft_gt="soft_datasets/soft_PascalVOC_2012_detection_trainval_fixed.json",
        run_name="voc_trainval_baseline_comp_simple",
        image_dirs=["VOC/images/train2012", "VOC/images/val2012"],
    ),
    "cityscapes_trainval": dict(
        parquet="matched_dataset/cityscapes_trainval_matched.parquet",
        soft_gt="soft_datasets/soft_Cityscapes_trainval_fixed.json",
        run_name="cityscapes_trainval_baseline_comp_simple",
        image_dirs=["Cityscapes/trainval/images"],
    ),
    "kitti_train": dict(
        parquet="matched_dataset/kitti_train_matched.parquet",
        soft_gt="soft_datasets/soft_Kitti_2D_train_fixed.json",
        run_name="kitti_train_baseline_comp_simple",
        image_dirs=["KITTI/train/images"],
    ),
}

SCORE_THR = 0.2
ONLY_OBJECTS = False
DATA_SPACE = "gdino_embedding"
AMGX_CONFIG = os.path.join(HERE, "../label_spreading/configs/FGMRES_AGGREGATION.json")
GRID_SEED = 42
BUDGET = 1.00   # 100% budget, per the AP-curve analysis

N_EXAMPLES = 4
MIN_BOX_FRAC = 0.02   # require at least ~2% of the image -- keeps examples legible at
                       # whole-image display scale; relaxed automatically per-bucket if
                       # that leaves too few candidates to pick N_EXAMPLES from.
UNCERTAIN_GT_MAX_PROB = 0.9   # "not entirely confident"
MAX_ASPECT = 4 / 3    # Cityscapes (2:1) and KITTI (3.3:1) images are cropped horizontally
                       # to COCO's typical aspect around the boxes -- at full width they fill
                       # the whole image panel and the bar plot's y-axis overlaps them.


def load_class_names(soft_gt_json, n_soft):
    """COCO's soft-GT lists only the real classes (background appended here, as in the
    notebook); VOC/Cityscapes/KITTI's already include the trailing `cantsolve` class,
    which plays the same background role, so it's renamed for consistent plots."""
    names = [c["name"] for c in soft_gt_json["categories"]]
    if len(names) == n_soft - 1:
        return names + ["background"]
    assert len(names) == n_soft and names[-1] == "cantsolve", (len(names), n_soft, names[-1])
    return names[:-1] + ["background"]


def top_k_legible(mask, conf, k, box_frac, min_frac=MIN_BOX_FRAC, label=""):
    """Top-k row indices by conf among mask, restricted to box_frac >= min_frac
    where possible -- halves min_frac until enough candidates exist (or gives up and
    uses the unfiltered mask), so a bucket with few legitimate candidates still fills
    rather than coming up short."""
    frac = min_frac
    while frac > 0:
        candidates = np.where(mask & (box_frac >= frac))[0]
        if len(candidates) >= k:
            if frac < min_frac:
                print(f"  [{label}] relaxed min_frac to {frac:.4f} ({len(candidates)} candidates)")
            return candidates[np.argsort(-conf[candidates])][:k]
        frac /= 2
    candidates = np.where(mask)[0]
    print(f"  [{label}] gave up on box_frac filter entirely ({len(candidates)} candidates)")
    return candidates[np.argsort(-conf[candidates])][:k]


def main(args):
    cfg = DATASETS[args.dataset]
    set_plot_layout(path_to_latex=args.latex_path)

    out_dir = os.path.join(HERE, "plots/examples")
    if args.dataset != "coco_val":
        out_dir = os.path.join(out_dir, args.dataset)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(args.data_dir, cfg["soft_gt"])) as f:
        soft_gt_json = json.load(f)
    IMAGE_FILENAME = {img["id"]: img["file_name"] for img in soft_gt_json["images"]}
    IMAGE_WH = {img["id"]: (img["width"], img["height"]) for img in soft_gt_json["images"]}
    image_dirs = [os.path.join(args.images_root, d) for d in cfg["image_dirs"]]

    df_full = pd.read_parquet(os.path.join(args.data_dir, cfg["parquet"]))
    dataset = df_full[df_full["gdino_score"] >= SCORE_THR].copy().reset_index(drop=True)
    sl = np.array([v for v in dataset["soft_label"].values])
    dataset["soft_label"] = list(sl)
    dataset["is_object"] = dataset["gdino_match"].astype(float)
    CLASS_NAMES = load_class_names(soft_gt_json, sl.shape[1])
    # box_frac: detection box area as a fraction of its image -- used to bias example
    # selection toward visually legible boxes
    img_wh = np.array([IMAGE_WH[iid] for iid in dataset["image_id"]])
    boxes = np.vstack(dataset["gdino_bbox"].to_numpy())
    box_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    dataset["box_frac"] = box_area / (img_wh[:, 0] * img_wh[:, 1])
    print(f"[{args.dataset}] {len(dataset):,} detections (score >= {SCORE_THR})")

    grid_results = pd.read_csv(os.path.join(
        args.experiments_dir, cfg["run_name"], "grid_search", f"{cfg['run_name']}_grid_search_results.csv"))

    def winning_hp(method, budget, metric="AP"):
        sub = grid_results[(grid_results["method"] == method) & (np.isclose(grid_results["budget"], budget))]
        return sub.loc[sub[metric].idxmax()]

    # ── estimates: calibrated GDino baseline, PLS, and PLS+Bayes ──────────────
    indices_b = sample_budget_indices(len(dataset), BUDGET, seed=GRID_SEED)
    n_samples_b = len(indices_b)

    tag = f"viz_{args.dataset}"
    baseline_ds, p_hat_baseline_b = calibrated_gdino_baseline(dataset, df_full, BUDGET, f"{tag}_baseline",
                                                                ONLY_OBJECTS, DATA_SPACE)

    bayes_row = winning_hp("Bayes", BUDGET)
    pls_ds = label_spreading_object_conventional(
        f"{tag}_pls", ONLY_OBJECTS, dataset.copy(), DATA_SPACE, "label", "soft_label", "is_object", "all",
        int(bayes_row["k"]), bayes_row["alpha"], n_samples_b, indices=indices_b, config=AMGX_CONFIG,
        bandwidth_scale=bayes_row["bandwidth_scale"],
    )
    pls_trials = pls_ds["trials"].to_numpy()
    pls_p_hat = np.vstack(pls_ds["p_hat"].to_numpy())
    F_pls = pls_p_hat * pls_trials[:, None]
    bayes_ds = build_bayes_dataset(bayes_row["kappa"], p_hat_baseline_b, F_pls, pls_ds, f"{tag}_bayes",
                                    int(bayes_row["k"]), bayes_row["alpha"], n_samples_b,
                                    ONLY_OBJECTS, DATA_SPACE, AMGX_CONFIG)
    print(f"Bayes winning combo (100% budget, best AP): k={int(bayes_row['k'])} alpha={bayes_row['alpha']} "
          f"bandwidth={bayes_row['bandwidth_scale']} kappa={bayes_row['kappa']}")

    our_p_hat = np.vstack(bayes_ds["p_hat"].to_numpy())
    gdino_p_hat = np.vstack(baseline_ds["p_hat"].to_numpy())
    gt_soft = np.vstack(dataset["soft_label"].to_numpy())

    our_pred = our_p_hat.argmax(axis=1)
    gdino_pred = gdino_p_hat.argmax(axis=1)
    gt_pred = gt_soft.argmax(axis=1)
    matched = dataset["gdino_match"].astype(bool).to_numpy()

    agree_mask = matched & (our_pred == gdino_pred)
    disagree_mask = matched & (our_pred != gdino_pred)
    print(f"Matched detections: {matched.sum():,}")
    print(f"  agreement (our argmax == GDino argmax):    {agree_mask.sum():,}")
    print(f"  disagreement (our argmax != GDino argmax): {disagree_mask.sum():,}")

    # ── example selection ─────────────────────────────────────────────────────
    box_frac = dataset["box_frac"].to_numpy()

    # Agreement: our_pred == gdino_pred == gt_pred, ranked by our confidence in that class
    agree_correct = agree_mask & (our_pred == gt_pred)
    agree_conf = our_p_hat[np.arange(len(dataset)), our_pred]
    agree_examples = top_k_legible(agree_correct, agree_conf, N_EXAMPLES, box_frac, label="agree")

    # Disagreement: prioritize "we fixed GDino's mistake" (our_pred == gt_pred != gdino_pred),
    # fill remaining slots with "GDino was right, we were wrong" for balance.
    we_fixed = disagree_mask & (our_pred == gt_pred) & (gdino_pred != gt_pred)
    gdino_right = disagree_mask & (gdino_pred == gt_pred) & (our_pred != gt_pred)
    we_fixed_conf = our_p_hat[np.arange(len(dataset)), our_pred]
    n_fixed_target = min(N_EXAMPLES - 1, int(we_fixed.sum()))
    we_fixed_sorted = top_k_legible(we_fixed, we_fixed_conf, max(n_fixed_target, 0), box_frac, label="we_fixed") \
        if n_fixed_target > 0 else np.array([], dtype=int)
    n_fixed = len(we_fixed_sorted)
    n_gdino_right = min(N_EXAMPLES - n_fixed, int(gdino_right.sum()))
    gdino_right_sorted = top_k_legible(gdino_right, we_fixed_conf, n_gdino_right, box_frac, label="gdino_right") \
        if n_gdino_right > 0 else np.array([], dtype=int)
    disagree_examples = np.concatenate([we_fixed_sorted, gdino_right_sorted]).astype(int)

    print(f"Agreement examples (row indices): {agree_examples.tolist()}")
    print(f"Disagreement examples (row indices): {disagree_examples.tolist()}  "
          f"({n_fixed} 'we fixed GDino', {len(gdino_right_sorted)} 'GDino was right')")

    gt_max_prob = gt_soft.max(axis=1)
    uncertain_gt_mask = matched & (gt_max_prob < UNCERTAIN_GT_MAX_PROB)
    print(f"Uncertain-GT candidates: {uncertain_gt_mask.sum():,} "
          f"({uncertain_gt_mask.sum() / max(matched.sum(), 1):.1%} of matched)")
    uncertain_gt_examples = top_k_legible(uncertain_gt_mask, -gt_max_prob, N_EXAMPLES, box_frac,
                                          label="uncertain_gt")
    print(f"Uncertain-GT examples (row indices): {uncertain_gt_examples.tolist()}")
    for idx in uncertain_gt_examples:
        top2 = np.argsort(-gt_soft[idx])[:2]
        top2_str = ", ".join(f"{CLASS_NAMES[c]}={gt_soft[idx][c]:.2f}" for c in top2)
        print(f"  row={idx}  gt_max={gt_max_prob[idx]:.3f}  top2: {top2_str}")

    # ── plotting ──────────────────────────────────────────────────────────────
    def find_image(file_name):
        for d in image_dirs:
            path = os.path.join(d, file_name)
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"{file_name} not found in any of {image_dirs}")

    def plot_example(row_idx, tag, save_path, top_k=4):
        row = dataset.iloc[row_idx]
        img = Image.open(find_image(IMAGE_FILENAME[int(row["image_id"])])).convert("RGB")

        det_x1, det_y1, det_x2, det_y2 = row["gdino_bbox"]
        gt_x, gt_y, gt_w, gt_h = row["bbox"]   # GT's own box geometry -- distinct from the detection's,
                                                # matched via IoU (gdino_bbox_iou), not the same coordinates

        # crop wide images to MAX_ASPECT, centered on both boxes (widened if the boxes need it)
        img_w, img_h = img.size
        if img_w > MAX_ASPECT * img_h:
            box_x1, box_x2 = min(det_x1, gt_x), max(det_x2, gt_x + gt_w)
            crop_w = min(img_w, max(MAX_ASPECT * img_h, (box_x2 - box_x1) * 1.1))
            crop_x1 = np.clip((box_x1 + box_x2) / 2 - crop_w / 2, 0, img_w - crop_w)
            img = img.crop((int(round(crop_x1)), 0, int(round(crop_x1 + crop_w)), img_h))
            det_x1, det_x2, gt_x = det_x1 - int(round(crop_x1)), det_x2 - int(round(crop_x1)), gt_x - int(round(crop_x1))

        gt, ours, gdino = gt_soft[row_idx], our_p_hat[row_idx], gdino_p_hat[row_idx]
        top_k = min(top_k, len(CLASS_NAMES))
        top_classes = np.argsort(-np.maximum.reduce([gt, ours, gdino]))[:top_k]
        labels = [CLASS_NAMES[c] for c in top_classes]

        fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(6.3, 2.4), gridspec_kw={"width_ratios": [1, 0.4], "wspace": 0.05})

        ax_img.imshow(img)
        ax_img.add_patch(Rectangle((det_x1, det_y1), det_x2 - det_x1, det_y2 - det_y1,
                                    fill=False, edgecolor="coral", linewidth=1.5))
        ax_img.add_patch(Rectangle((gt_x, gt_y), gt_w, gt_h,
                                    fill=False, edgecolor="teal", linewidth=1.5))
        ax_img.axis("off")
        # Line2D proxies instead of the Rectangle patches themselves -- a small hollow
        # rectangle icon reads poorly as a legend swatch once the real boxes are small.
        legend_handles = [Line2D([0], [0], color="coral", lw=1.5, label="Detection"),
                           Line2D([0], [0], color="teal", lw=1.5, label="GT")]
        ax_img.legend(handles=legend_handles, fontsize=6, frameon=True, facecolor="white",
                      framealpha=0.85, edgecolor="none", loc="upper right",
                      handlelength=1.3, borderaxespad=0.2, labelspacing=0.2)

        x = np.arange(top_k)
        width = 0.27
        ax_bar.bar(x - width, gt[top_classes], width, label="Soft GT", color="slategrey")
        ax_bar.bar(x, ours[top_classes], width, label="Bayesian PLS", color="teal")
        ax_bar.bar(x + width, gdino[top_classes], width, label="Grounding DINO", color="coral")
        ax_bar.set_xticks(x)
        ax_bar.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax_bar.tick_params(axis="y", labelsize=7)
        ax_bar.set_ylabel("Probability", fontsize=8, labelpad=3)
        ax_bar.set_ylim(0, 1)
        ax_bar.legend(fontsize=6, frameon=False, loc="upper right", handlelength=1.3, borderaxespad=0.2,
                      labelspacing=0.2)
        ax_bar.grid(True, axis="y", ls=":", alpha=0.6)

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [{tag}] image_id={row['image_id']} row={row_idx}  "
              f"GT={CLASS_NAMES[gt_pred[row_idx]]}  Ours={CLASS_NAMES[our_pred[row_idx]]}  "
              f"GDino={CLASS_NAMES[gdino_pred[row_idx]]}")

    for name, examples in [("agreement", agree_examples), ("disagreement", disagree_examples),
                           ("uncertain_gt", uncertain_gt_examples)]:
        for i, idx in enumerate(examples):
            plot_example(idx, f"{name}-{i}", os.path.join(out_dir, f"{name}_{i}.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--data_dir", default=os.path.join(HERE, "../data"))
    parser.add_argument("--experiments_dir", default=os.path.join(HERE, "experiments"))
    parser.add_argument("--images_root", default="/home/klees/data")
    parser.add_argument("--latex_path", default="/home/klees/texlive/2024/bin/x86_64-linux")
    main(parser.parse_args())
