"""Shared helpers for the label-error-detection study, used by
define_label_errors.ipynb (builds the label-error ground truth) and
parameter_sweep.py (scores detectors' ability to recover it).

Matching/AP methodology is deliberately ported from the existing reference
implementation https://github.com/JonathanKlees/Uncertainty-aware-Object-Detection
(not reinvented): `greedy_match`/`compute_iou` mirrors
labelerrors/helpers_legt.py's GT-vs-GT matching (global best-IoU-first,
category-agnostic, threshold 0.5); `compute_pr_curve`/`compute_ap` mirrors
labelerrordetectionbenchmark/precision_recall_curve/prec_rec_curve_all_methods.py's
"Score baseline" method (score-ordered, per-image greedy best-IoU-among-
unmatched, threshold 0.1, standard all-points AP).

All boxes handled here are XYXY unless a function name says otherwise
(`*_xywh`) -- the original/soft GT jsons and gdino_bbox are all naturally
XYXY already except the original hard-GT COCO-format jsons, which store
`bbox` as COCO XYWH and get converted on load.
"""
import json

import numpy as np
import pandas as pd
import torch
from torchvision.ops import nms


# ── Box / IoU primitives ──────────────────────────────────────────────────────
def xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]


def compute_iou(box_a, box_b):
    """Single-pair IoU, XYXY boxes."""
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])
    inter = max(0.0, xb - xa) * max(0.0, yb - ya)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(boxes_a, boxes_b):
    """(N,4) x (M,4) XYXY -> (N,M) IoU matrix, vectorized."""
    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    x1 = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
    y1 = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
    x2 = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
    y2 = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-9)


# ── Detector-side pre-processing: NMS ──────────────────────────────────────────
def apply_nms(det_df, iou_threshold=0.6, bbox_col="gdino_bbox", score_col="gdino_score",
              image_id_col="image_id", label_col="label"):
    """Per-(image, predicted class) NMS via torchvision.ops.nms -- same convention
    as le_benchmark/tools_label_error_detection.py's perform_nms_on_dataframe
    (grouped by filename+class, not category-agnostic). Collapses the flood of
    near-duplicate boxes a detector emits at low score thresholds, which matters
    especially once SCORE_THR is pushed very low (e.g. 0.01)."""
    kept_parts = []
    for (_, _), group in det_df.groupby([image_id_col, label_col]):
        boxes = torch.tensor(np.stack(group[bbox_col].to_numpy()), dtype=torch.float32)
        scores = torch.tensor(group[score_col].to_numpy(), dtype=torch.float32)
        keep = nms(boxes, scores, iou_threshold)
        kept_parts.append(group.iloc[keep.numpy()])
    return pd.concat(kept_parts).sort_index().reset_index(drop=True)


# ── Label-error GT: original GT vs. soft-GT (thresholded to hard) ─────────────
def greedy_match_by_iou(boxes_a, boxes_b, iou_threshold=0.5):
    """GT-vs-GT greedy matching: global best-IoU-first (NOT score-ordered --
    there's no score for either side here), category-agnostic, one-to-one.
    Mirrors helpers_legt.py's greedy_match exactly. Returns (matches, iou_mat)
    where matches is a list of (i, j, iou) and iou_mat is the full (N,M) matrix
    (handed back so callers can also get each unmatched box's max-IoU, as
    helpers_legt.py's "missing"/"misaligned" sub-typing does)."""
    iou_mat = iou_matrix(boxes_a, boxes_b)
    n, m = iou_mat.shape
    pairs = [(iou_mat[i, j], i, j) for i in range(n) for j in range(m)]
    pairs.sort(key=lambda t: t[0], reverse=True)

    used_a, used_b = set(), set()
    matches = []
    for iou, i, j in pairs:
        if iou < iou_threshold:
            break   # sorted descending -- everything after this is also below threshold
        if i in used_a or j in used_b:
            continue
        matches.append((i, j, iou))
        used_a.add(i)
        used_b.add(j)
    return matches, iou_mat


def soft_gt_to_hard_boxes(soft_gt_path, filename_to_image_id, soft_label_thresh=0.5):
    """Per-image list of (bbox_xyxy, class_idx, prob) for every soft-GT object
    whose max non-background class probability exceeds soft_label_thresh --
    same thresholding rule as label_spreading/evaluation.py's
    _load_hard_gt_from_soft (soft_label[:-1] drops the trailing cantsolve/
    background entry before taking argmax), reimplemented here to return
    per-image box lists directly instead of a pycocotools COCO object, which
    is what the greedy-matching step below needs."""
    with open(soft_gt_path) as f:
        soft_gt = json.load(f)

    by_image = {}
    for file_name, objects in soft_gt["objects"].items():
        image_id = filename_to_image_id.get(file_name)
        if image_id is None:
            continue
        kept = []
        for obj in objects:
            soft_label = obj["soft_label"][:-1]
            cls = int(np.argmax(soft_label))
            prob = soft_label[cls]
            if prob <= soft_label_thresh:
                continue
            kept.append({"bbox": list(obj["bbox"]), "class_idx": cls, "prob": float(prob)})
        by_image[image_id] = kept
    return by_image


def original_gt_to_hard_boxes(original_gt_json):
    """Per-image list of (bbox_xyxy, category_id) from a standard COCO-format
    hard-label GT json (the dataset's own original annotations)."""
    images_by_fname = {img["file_name"]: img["id"] for img in original_gt_json["images"]}
    by_image = {img["id"]: [] for img in original_gt_json["images"]}
    for ann in original_gt_json["annotations"]:
        image_id = ann["image_id"]
        if image_id not in by_image:
            continue
        by_image[image_id].append({"bbox": xywh_to_xyxy(ann["bbox"]), "category_id": ann["category_id"]})
    return by_image, images_by_fname


def find_label_errors(original_by_image, soft_hard_by_image, iou_threshold=0.5):
    """Label-error GT construction: for every image, greedy-match original-GT boxes against
    hard-thresholded soft-GT boxes (category-agnostic, IoU>=iou_threshold).
    Every UNMATCHED soft-GT box is a label error (an object the original
    annotators missed entirely or too badly to count as the same box) --
    the union of helpers_legt.py's "missing" + "misaligned" categories,
    not sub-typed further since this study doesn't need that distinction.

    Returns a flat list of label-error dicts: {image_id, bbox (xyxy),
    class_idx, prob, max_iou_with_original}."""
    label_errors = []
    all_image_ids = set(original_by_image) | set(soft_hard_by_image)
    for image_id in all_image_ids:
        orig_boxes = [o["bbox"] for o in original_by_image.get(image_id, [])]
        soft_objs = soft_hard_by_image.get(image_id, [])
        soft_boxes = [o["bbox"] for o in soft_objs]

        if len(soft_objs) == 0:
            continue
        if len(orig_boxes) == 0:
            # nothing in the original at all -- every soft-GT box here is a miss
            for obj in soft_objs:
                label_errors.append(dict(image_id=image_id, bbox=obj["bbox"], class_idx=obj["class_idx"],
                                          prob=obj["prob"], max_iou_with_original=0.0))
            continue

        matches, iou_mat = greedy_match_by_iou(orig_boxes, soft_boxes, iou_threshold=iou_threshold)
        matched_soft = {j for _, j, _ in matches}
        max_iou_per_soft = iou_mat.max(axis=0) if iou_mat.size else np.zeros(len(soft_boxes))

        for j, obj in enumerate(soft_objs):
            if j in matched_soft:
                continue
            label_errors.append(dict(image_id=image_id, bbox=obj["bbox"], class_idx=obj["class_idx"],
                                      prob=obj["prob"], max_iou_with_original=float(max_iou_per_soft[j])))
    return label_errors


def find_label_errors_reference_consistent(original_by_image, soft_hard_by_image, catid_to_idx,
                                            iou_threshold=0.5, classification_iou_threshold=0.7):
    """Same idea as find_label_errors, but matching Uncertainty-aware-Object-Detection's
    helpers_legt.py::extract_label_errors exactly, fixing two discrepancies found by
    comparing against it directly:
      1. Unmatched soft-GT boxes are only kept as label errors ("missing"/"misaligned")
         if their max IoU against original GT is < iou_threshold. find_label_errors
         above keeps ALL unmatched boxes regardless of max_iou -- the reference drops
         the rare case of a box that has high IoU (>=iou_threshold) against its best
         original-GT candidate but still ended up unmatched (lost a greedy-matching
         collision to a different soft-GT box with even higher IoU to that same
         original box). Confirmed empirically to be a small effect (~1.7% of COCO's
         label errors).
      2. A third error type -- "classification" -- is added: a MATCHED pair (from the
         same greedy match used for missing/misaligned) with IoU >= classification_iou_threshold
         but a DIFFERENT class between original and soft-GT. find_label_errors doesn't
         track this at all (out of scope for its "object present/absent" framing) --
         these are geometrically-correct boxes with a wrong label, not missed objects.

    catid_to_idx: maps original-GT category_id -> the soft-label class_idx space (they
    share the same id scheme up to the soft side's extra trailing background/cantsolve
    entry, verified across all 4 datasets in this project).

    Returns the same flat list-of-dicts shape as find_label_errors, plus a "type" key
    ("missing"/"misaligned"/"classification") for reference."""
    label_errors = []
    all_image_ids = set(original_by_image) | set(soft_hard_by_image)
    for image_id in all_image_ids:
        orig_objs = original_by_image.get(image_id, [])
        orig_boxes = [o["bbox"] for o in orig_objs]
        soft_objs = soft_hard_by_image.get(image_id, [])
        soft_boxes = [o["bbox"] for o in soft_objs]

        if len(soft_objs) == 0:
            continue
        if len(orig_boxes) == 0:
            for obj in soft_objs:
                label_errors.append(dict(image_id=image_id, bbox=obj["bbox"], class_idx=obj["class_idx"],
                                          prob=obj["prob"], max_iou_with_original=0.0, type="missing"))
            continue

        matches, iou_mat = greedy_match_by_iou(orig_boxes, soft_boxes, iou_threshold=iou_threshold)
        matched_soft = {j for _, j, _ in matches}
        max_iou_per_soft = iou_mat.max(axis=0) if iou_mat.size else np.zeros(len(soft_boxes))

        for j, obj in enumerate(soft_objs):
            if j in matched_soft:
                continue
            iou = float(max_iou_per_soft[j])
            if iou < iou_threshold:
                err_type = "missing" if iou <= 0.1 else "misaligned"
                label_errors.append(dict(image_id=image_id, bbox=obj["bbox"], class_idx=obj["class_idx"],
                                          prob=obj["prob"], max_iou_with_original=iou, type=err_type))
            # else: iou >= iou_threshold but unmatched (lost a greedy collision) -- dropped

        for i, j, iou in matches:
            if iou < classification_iou_threshold:
                continue
            orig_idx = catid_to_idx.get(orig_objs[i]["category_id"])
            soft_idx = soft_objs[j]["class_idx"]
            if orig_idx is not None and orig_idx != soft_idx:
                label_errors.append(dict(image_id=image_id, bbox=soft_objs[j]["bbox"], class_idx=soft_idx,
                                          prob=soft_objs[j]["prob"], max_iou_with_original=float(iou),
                                          type="classification"))
    return label_errors


def build_catid_to_idx(soft_gt_path):
    """Maps original-GT category_id -> soft-label class_idx, from the soft-GT json's
    own "categories" list (ordered to match class_idx position; shares original GT's
    id scheme up to soft's extra trailing background/cantsolve entry)."""
    with open(soft_gt_path) as f:
        soft_gt = json.load(f)
    idx_to_catid = {i: cat["id"] for i, cat in enumerate(soft_gt["categories"])}
    return {catid: idx for idx, catid in idx_to_catid.items()}


# ── Detections vs. original GT -> "originally FP" proposals ───────────────────
def match_detections_to_gt(det_df, gt_by_image, iou_threshold=0.5,
                            bbox_col="gdino_bbox", score_col="gdino_score", image_id_col="image_id"):
    """Standard score-ordered greedy detection-vs-GT matching (same convention
    as build_matching_dataset.py's greedy_match: sort by confidence descending,
    per image assign each detection to its best still-unmatched GT box if
    IoU>=iou_threshold), category-agnostic here since the rest of this study
    is too. Returns a boolean array (same order as det_df) -- True if the
    detection matched some ORIGINAL-GT box (i.e. is NOT a candidate label-error
    proposal), False otherwise."""
    order = det_df[score_col].to_numpy().argsort()[::-1]
    matched = np.zeros(len(det_df), dtype=bool)

    used_gt_by_image = {}
    boxes_by_image_idx = {}
    for image_id, sub in det_df.groupby(image_id_col):
        boxes_by_image_idx[image_id] = {}

    image_ids = det_df[image_id_col].to_numpy()
    bboxes = det_df[bbox_col].to_numpy()

    used_gt = {image_id: np.zeros(len(gt_by_image.get(image_id, [])), dtype=bool) for image_id in gt_by_image}

    for idx in order:
        image_id = image_ids[idx]
        gt_objs = gt_by_image.get(image_id, [])
        if len(gt_objs) == 0:
            continue
        gt_boxes = [o["bbox"] for o in gt_objs]
        ious = np.array([compute_iou(bboxes[idx], gb) for gb in gt_boxes])
        used = used_gt.setdefault(image_id, np.zeros(len(gt_objs), dtype=bool))
        ious[used] = -1.0
        best = int(np.argmax(ious))
        if ious[best] >= iou_threshold:
            matched[idx] = True
            used[best] = True
    return matched


# ── Proposals vs. label errors -> PR curve / AP / achieved recall ─────────────
def compute_pr_curve(proposals, label_errors, iou_thresh=0.1):
    """Score-ordered, per-image greedy best-IoU-among-unmatched matching of
    `proposals` (list of {image_id, bbox, score}) against `label_errors` (list
    of {image_id, bbox}) -- exact port of
    labelerrordetectionbenchmark/.../prec_rec_curve_all_methods.py's
    compute_pr_curve, at this study's IoU=0.1. Returns (precision, recall,
    scores) arrays, one entry per proposal in score-descending order."""
    total_gt = len(label_errors)
    if total_gt == 0:
        raise ValueError("No label errors found for this dataset.")

    gt_by_image = {}
    for le in label_errors:
        gt_by_image.setdefault(le["image_id"], []).append({"bbox": le["bbox"], "matched": False})

    proposals = sorted(proposals, key=lambda p: p["score"], reverse=True)

    tp_list, fp_list, scores = [], [], []
    for p in proposals:
        gt_boxes = gt_by_image.get(p["image_id"], [])
        best_iou, best_gt = 0.0, None
        for gt in gt_boxes:
            if gt["matched"]:
                continue
            iou = compute_iou(p["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou, best_gt = iou, gt
        if best_iou >= iou_thresh and best_gt is not None:
            tp_list.append(1)
            fp_list.append(0)
            best_gt["matched"] = True
        else:
            tp_list.append(0)
            fp_list.append(1)
        scores.append(p["score"])

    tp_cum = np.cumsum(tp_list)
    fp_cum = np.cumsum(fp_list)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    recall = tp_cum / total_gt
    return precision, recall, np.asarray(scores)


def compute_ap(precision, recall):
    """Standard all-points AP (precision envelope + rectangle integration under
    the interpolated PR curve) -- same as prec_rec_curve_all_methods.py's
    compute_ap / the PASCAL VOC all-points definition."""
    precision = np.asarray(precision)
    recall = np.asarray(recall)
    if len(precision) == 0:
        return 0.0
    mrec = np.concatenate(([0.0], recall))
    mpre = np.concatenate(([precision[0]], precision))
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    return float(np.sum((mrec[1:] - mrec[:-1]) * mpre[1:]))


def random_baseline(proposals, label_errors, iou_thresh=0.1, seed=0):
    """Closed-form AP/F1 for a classifier carrying zero information: a flat PR
    curve at precision = (matches found using every proposal) / (proposal
    count), over the full recall range it reaches -- AP for a flat segment is
    just precision * recall (base * height of the rectangle under it).
    Implemented by running the same compute_pr_curve matcher with each
    proposal's score REPLACED by an iid random draw (compute_pr_curve sorts by
    each proposal's own "score" field internally, so merely permuting list
    order without doing this would just be re-sorted straight back to the
    real ranking). The per-image greedy matching is order-dependent when
    several proposals in one image compete for the same box, so this random
    ranking is a faithful 'no-skill' sample rather than reusing any real
    score's ordering. Returns the same dict shape as the evaluate_score()
    closures used elsewhere in this study, so it drops in as just another
    method."""
    rng = np.random.default_rng(seed)
    random_scores = rng.uniform(size=len(proposals))
    shuffled = [{**p, "score": s} for p, s in zip(proposals, random_scores)]
    precision, recall, _ = compute_pr_curve(shuffled, label_errors, iou_thresh=iou_thresh)
    p_all = float(precision[-1]) if len(precision) else 0.0
    r_all = float(recall[-1]) if len(recall) else 0.0
    ap = p_all * r_all
    f1 = 2 * p_all * r_all / (p_all + r_all) if (p_all + r_all) > 0 else 0.0
    return dict(precision=precision, recall=recall, ap=ap, achieved_recall=r_all,
                f1=f1, f1_precision=p_all, f1_recall=r_all)
