"""Helpers for calibrating OVOD detection scores: load detections, score each one as
TP/FP against COCO-format ground truth (via a COCOeval run), and plot reliability
diagrams with the expected calibration error (ECE)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def gather_detections_from_parquet(image_ids, parquet_path):
    """Load detections from a parquet file and return in COCO detection format.

    The parquet schema produced by gdino_inference_coco.py has columns:
      image_id, score, label, box ([x1,y1,x2,y2]), probs (array-91), last_hidden_state.
    Boxes are converted to COCO format [x,y,w,h], ready for COCO.loadRes.
    """
    id_set = set(int(i) for i in image_ids)
    df = pd.read_parquet(
        parquet_path,
        columns=["image_id", "label", "score", "box", "probs"]
    )
    df = df[df["image_id"].isin(id_set)].reset_index(drop=True)

    if df.empty:
        return []

    # Vectorised box conversion [x1,y1,x2,y2] → COCO [x,y,w,h]
    boxes = np.array(df["box"].tolist(), dtype=np.float32)
    coco_boxes = boxes.copy()
    coco_boxes[:, 2] -= coco_boxes[:, 0]   # w = x2 - x1
    coco_boxes[:, 3] -= coco_boxes[:, 1]   # h = y2 - y1

    detections = []
    for i, row in enumerate(df.itertuples(index=False)):
        probs = row.probs
        if hasattr(probs, "tolist"):
            probs = probs.tolist()
        detections.append({
            "image_id"   : int(row.image_id),
            "category_id": int(row.label),
            "bbox"       : coco_boxes[i].tolist(),
            "score"      : float(row.score),
            "probs"      : probs,
        })
    return detections


def extract_per_image_scores(cocoEval):
    """Per-image detection scores and TP labels (1 = matched, not ignored) from an
    evaluated COCOeval (first IoU threshold)."""
    per_image = {}

    for eval_img in cocoEval.evalImgs:
        if eval_img is None:
            continue

        image_id = eval_img["image_id"]

        scores = np.asarray(eval_img["dtScores"])
        matches = eval_img["dtMatches"][0]
        ignore = eval_img["dtIgnore"][0]

        labels = ((matches > 0) & (~ignore)).astype(int)

        if image_id not in per_image:
            per_image[image_id] = {
                "scores": [],
                "labels": []
            }

        per_image[image_id]["scores"].append(scores)
        per_image[image_id]["labels"].append(labels)

    for image_id in per_image:
        per_image[image_id]["scores"] = np.concatenate(
            per_image[image_id]["scores"]
        )
        per_image[image_id]["labels"] = np.concatenate(
            per_image[image_id]["labels"]
        )

    return per_image


def flatten_per_image_data(per_image_data):
    scores_all = []
    labels_all = []
    image_ids_all = []

    for img_id, data in per_image_data.items():
        n = len(data["scores"])

        scores_all.append(data["scores"])
        labels_all.append(data["labels"])
        image_ids_all.append(np.full(n, img_id))

    scores_all = np.concatenate(scores_all)
    labels_all = np.concatenate(labels_all)
    image_ids_all = np.concatenate(image_ids_all)

    return scores_all, labels_all, image_ids_all


def compute_ece(scores, labels, n_bins=10):
    # Expected Calibration Error (ECE)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(scores, bins) - 1

    ece = 0.0
    N = len(scores)

    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue

        bin_acc = np.mean(labels[mask])
        bin_conf = np.mean(scores[mask])
        ece += (np.sum(mask) / N) * abs(bin_acc - bin_conf)

    return ece


def plot_reliability_diagram(scores, labels, n_bins=10, filename = None):
    # Calibration Plot (Reliability Diagram) with ECE annotation

    ece = compute_ece(scores, labels, n_bins)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(scores, bins) - 1
    bin_means = (bins[:-1] + bins[1:]) / 2

    bin_accs = []
    bin_confs = []

    for b in range(n_bins):
        mask = bin_ids == b
        if not np.any(mask):
            continue

        bin_accs.append(np.mean(labels[mask]))
        bin_confs.append(np.mean(scores[mask]))

    plt.bar(bin_means[:len(bin_accs)], bin_accs, width=0.1, ec = "slategray", align="center", label="Accuracy")
    plt.plot([0, 1], [0, 1], linestyle="--", lw = 1.5, color="gray")
    plt.xlabel("Mean Confidence")
    plt.ylabel("Mean Accuracy")
    plt.annotate(
        'ECE: %.4f' % ece,
        xy=(0.05, 0.85), xycoords='axes fraction',
        bbox=dict(boxstyle="round", fc="0.9", ec="teal")
    )
    if filename:
        plt.savefig(filename, bbox_inches="tight", dpi = 300)
    plt.show()
    plt.close()
