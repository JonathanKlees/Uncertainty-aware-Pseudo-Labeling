import io
import contextlib
import json
from collections import defaultdict

import numpy as np
import pandas as pd
import sklearn.metrics
import torch

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


_SOFT_GT_CACHE = {}
_SOFT_GT_BY_IMAGE_CACHE = {}


def _load_hard_gt_from_soft(soft_gt_path, soft_label_thresh=0.5):
    """Loads a soft-GT json (per-object soft_label distributions over the 80 COCO
    classes + a trailing 'cantsolve'/background entry, as produced by the
    label-spreading pipeline's soft-GT track, e.g. soft_COCO_2017_val.json) and
    converts it to a hard-label COCO-format ground truth by thresholding: an
    annotation is only kept if its soft label's max class probability (ignoring
    the trailing cantsolve entry) exceeds `soft_label_thresh`, taking the argmax
    class as its single hard label. Cached per (path, thresh) since the same GT is
    typically evaluated against many methods/hyperparameters in a row.

    Returns a ready-to-use pycocotools.COCO object."""
    cache_key = (soft_gt_path, soft_label_thresh)
    if cache_key in _SOFT_GT_CACHE:
        return _SOFT_GT_CACHE[cache_key]

    with open(soft_gt_path) as f:
        soft_gt = json.load(f)

    categories = soft_gt["categories"]
    id_map = {i: cat["id"] for i, cat in enumerate(categories)}   # soft_label index -> COCO category_id
    filename_to_id = {img["file_name"]: img["id"] for img in soft_gt["images"]}

    annotations = []
    ann_id = 1
    for file_name, objects in soft_gt["objects"].items():
        image_id = filename_to_id[file_name]
        for obj in objects:
            soft_label = obj["soft_label"][:-1]   # drop the trailing cantsolve/background entry
            cls = int(np.argmax(soft_label))
            prob = soft_label[cls]
            if prob <= soft_label_thresh:
                continue
            x1, y1, x2, y2 = obj["bbox"]
            w, h = x2 - x1, y2 - y1
            annotations.append({
                "id": ann_id, "image_id": image_id, "category_id": id_map[cls],
                "bbox": [x1, y1, w, h], "area": float(w * h), "iscrowd": 0,
            })
            ann_id += 1

    coco_gt = COCO()
    coco_gt.dataset = {"info": {}, "images": soft_gt["images"], "annotations": annotations, "categories": categories}
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt.createIndex()

    _SOFT_GT_CACHE[cache_key] = coco_gt
    return coco_gt


def compute_map(process_data, soft_gt_path="../data/soft_datasets/soft_COCO_2017_val.json",
                 soft_label_thresh=0.5, verbose=False, scores=None, return_curve=False, pr_curve_iou=None):
    """Computes COCO mAP/AR metrics from `process_data` with pycocotools, against
    the soft GT thresholded into hard labels (see _load_hard_gt_from_soft):

    - ground truth: every soft-GT annotation whose max class probability exceeds
      `soft_label_thresh`, hard-labeled as that argmax class -- the full,
      independent COCO val annotation set, not just whatever this particular
      `process_data` run happened to match.
    - detections: the *original* GroundingDINO proposals (`gdino_bbox`,
      `gdino_label`), re-scored instead of the raw `gdino_score` -- i.e. this
      measures whether re-ranking the same boxes/categories by the propagated
      estimate improves detection quality.

    By default (`scores=None`) ranks by the spread objectness estimate
    `score_est[:, 1]` (p(object present), from the TP/FP-label diffusion --
    `score_est[:, 0]` is redundant, just 1 - score_est[:, 1]). Pass `scores`
    (array-like, same length/row order as `process_data`) to rank by a
    different confidence instead -- e.g. the soft-label diffusion's own max
    non-background class probability, to compare a TP/FP-based objectness
    score against a purely soft-label-based one.

    Returns a dict of the 12 standard COCOeval stats: AP, AP50, AP75, AP_small,
    AP_medium, AP_large, AR_1, AR_10, AR_100, AR_small, AR_medium, AR_large.

    If `return_curve=True`, also adds "AP_curve": a {iou_thresh: AP} dict for
    all 10 standard COCO IoU thresholds (0.5:0.95), area='all', maxDets=100 --
    i.e. the per-threshold breakdown "AP" itself already averages over
    (AP50/AP75 are just this curve's values at two of the ten points), read
    directly off `ev.eval['precision']` the same way COCOeval's own
    `summarize()` computes AP50/AP75 internally, no extra evaluation pass.

    If `pr_curve_iou` is set (e.g. 0.5), also adds "AP_pr_curve":
    {"recall": [101 values, 0..1, COCOeval's own fixed recall grid], "precision":
    [101 values]} -- the actual precision-vs-recall curve at that single IoU
    threshold, area='all', maxDets=100, averaged over categories (COCOeval
    computes precision per-category; unlike AS this must be explicitly
    averaged to get one curve, since AP is a per-class metric to begin with)."""
    coco_gt = _load_hard_gt_from_soft(soft_gt_path, soft_label_thresh)
    name_to_id = {cat["name"]: cat["id"] for cat in coco_gt.dataset["categories"]}

    cat_ids = process_data["gdino_label"].map(name_to_id)
    valid = cat_ids.notna().values
    if not valid.any():
        raise ValueError("compute_map: no gdino_label in process_data resolves to a known "
                          "COCO category name")

    boxes_xyxy = np.vstack(process_data.loc[valid, "gdino_bbox"].to_numpy()).astype(float)
    boxes_xywh = boxes_xyxy.copy()
    boxes_xywh[:, 2] -= boxes_xywh[:, 0]
    boxes_xywh[:, 3] -= boxes_xywh[:, 1]

    if scores is None:
        scores = np.vstack(process_data.loc[valid, "score_est"].to_numpy())[:, 1]   # p(object present)
    else:
        scores = np.asarray(scores)[valid]

    detections = [
        {"image_id": int(image_id), "category_id": int(cat_id), "bbox": box.tolist(), "score": float(score)}
        for image_id, cat_id, box, score in zip(
            process_data.loc[valid, "image_id"], cat_ids[valid].astype(int), boxes_xywh, scores)
    ]

    with contextlib.redirect_stdout(io.StringIO()):
        coco_dt = coco_gt.loadRes(detections)
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.evaluate()
        ev.accumulate()

    if verbose:
        ev.summarize()
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            ev.summarize()

    stat_names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
                  "AR_1", "AR_10", "AR_100", "AR_small", "AR_medium", "AR_large"]
    result = dict(zip(stat_names, ev.stats))

    if return_curve:
        precision = ev.eval["precision"]   # [T, R, K, A, M]
        aind = [i for i, a in enumerate(ev.params.areaRngLbl) if a == "all"]
        mind = [i for i, m in enumerate(ev.params.maxDets) if m == 100]
        ap_curve = {}
        for iou_thresh in ev.params.iouThrs:
            t = np.where(np.isclose(iou_thresh, ev.params.iouThrs))[0]
            s = precision[t][:, :, :, aind, mind]
            ap_curve[round(float(iou_thresh), 2)] = float(np.mean(s[s > -1])) if (s > -1).any() else float("nan")
        result["AP_curve"] = ap_curve

    if pr_curve_iou is not None:
        precision = ev.eval["precision"]   # [T, R, K, A, M]
        aind = [i for i, a in enumerate(ev.params.areaRngLbl) if a == "all"][0]
        mind = [i for i, m in enumerate(ev.params.maxDets) if m == 100][0]
        t_idx = np.where(np.isclose(pr_curve_iou, ev.params.iouThrs))[0]
        if len(t_idx) == 0:
            raise ValueError(f"pr_curve_iou={pr_curve_iou} is not one of COCOeval's standard "
                              f"IoU thresholds {ev.params.iouThrs.tolist()}")
        pr_slice = precision[t_idx[0], :, :, aind, mind]   # [R, K] -- K = per-category precision at each recall level
        precision_curve = np.full(pr_slice.shape[0], np.nan)
        for r in range(pr_slice.shape[0]):
            valid = pr_slice[r] > -1
            if valid.any():
                precision_curve[r] = pr_slice[r][valid].mean()   # average over categories, this recall level only
        result["AP_pr_curve"] = {"recall": ev.params.recThrs.tolist(), "precision": precision_curve.tolist()}

    return result


def _load_soft_gt_by_image(soft_gt_path):
    """Loads a soft-GT json (see _load_hard_gt_from_soft) and groups every object's
    (xyxy bbox, 81-dim soft_label) by image_id, without thresholding -- used by
    compute_similarity, which compares full soft-label distributions rather than
    hard-thresholded classes. Cached per path."""
    if soft_gt_path in _SOFT_GT_BY_IMAGE_CACHE:
        return _SOFT_GT_BY_IMAGE_CACHE[soft_gt_path]

    with open(soft_gt_path) as f:
        soft_gt = json.load(f)

    filename_to_id = {img["file_name"]: img["id"] for img in soft_gt["images"]}

    raw_by_image = defaultdict(lambda: {"boxes": [], "soft_labels": []})
    for file_name, objects in soft_gt["objects"].items():
        image_id = filename_to_id[file_name]
        for obj in objects:
            raw_by_image[image_id]["boxes"].append(obj["bbox"])
            raw_by_image[image_id]["soft_labels"].append(obj["soft_label"])

    gt_by_image = {
        image_id: {
            "boxes": np.array(v["boxes"], dtype=np.float32),
            "soft_labels": np.array(v["soft_labels"], dtype=np.float32),
        }
        for image_id, v in raw_by_image.items()
    }
    _SOFT_GT_BY_IMAGE_CACHE[soft_gt_path] = gt_by_image
    return gt_by_image


def _iou_matrix(boxes1, boxes2):
    """boxes1 (N,4), boxes2 (M,4): xyxy tensors. Returns the (N, M) IoU matrix."""
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-8)


def _coco_style_matching(iou_matrix, scores, pred_areas, gt_areas, iou_thresh, area_range):
    """Greedy COCO-style 1:1 matching (mirrors pycocotools' COCOeval.evaluateImg,
    but category-agnostic -- soft labels carry the class uncertainty, so there's
    no discrete GT category to match on): predictions sorted by score, each
    matched to its best-IoU (>= iou_thresh) unused GT, preferring non-ignored GTs
    (outside area_range) but falling back to an ignored one if that's the only
    match above threshold. Returns matched_pred (index into GT, or -1), dt_ignore,
    gt_ignore (both bool), one entry per prediction/GT respectively."""
    area_min, area_max = area_range
    device = iou_matrix.device
    n_pred, n_gt = iou_matrix.shape

    if n_gt == 0:
        matched_pred = torch.full((n_pred,), -1, dtype=torch.long, device=device)
        dt_ignore = (pred_areas < area_min) | (pred_areas > area_max)
        gt_ignore = torch.zeros((0,), dtype=torch.bool, device=device)
        return matched_pred, dt_ignore, gt_ignore

    gt_ignore = (gt_areas < area_min) | (gt_areas > area_max)

    iou_valid = iou_matrix.clone()
    iou_valid[:, gt_ignore] = -1   # prefer non-ignored GTs first
    best_iou, best_gt = iou_valid.max(dim=1)

    fallback_mask = best_iou < iou_thresh   # no valid (non-ignored) GT found -> allow an ignored one
    if fallback_mask.any():
        iou_fallback = iou_matrix[fallback_mask]
        best_iou_fb, best_gt_fb = iou_fallback.max(dim=1)
        use_fb = best_iou_fb >= iou_thresh
        best_gt[fallback_mask] = torch.where(use_fb, best_gt_fb, best_gt[fallback_mask])
        best_iou[fallback_mask] = torch.where(use_fb, best_iou_fb, best_iou[fallback_mask])

    valid = best_iou >= iou_thresh
    pred_idx = torch.arange(n_pred, device=device)[valid]
    gt_idx = best_gt[valid]
    pred_scores = scores[valid]

    order = torch.argsort(pred_scores, descending=True)
    pred_idx, gt_idx = pred_idx[order], gt_idx[order]

    matched_pred = torch.full((n_pred,), -1, dtype=torch.long, device=device)
    used_gt = torch.zeros(n_gt, dtype=torch.bool, device=device)
    for i in range(len(pred_idx)):
        p, g = pred_idx[i], gt_idx[i]
        if not used_gt[g]:
            matched_pred[p] = g
            used_gt[g] = True

    matched_mask = matched_pred >= 0
    dt_ignore = torch.zeros(n_pred, dtype=torch.bool, device=device)
    dt_ignore[matched_mask] = gt_ignore[matched_pred[matched_mask]]
    unmatched = ~matched_mask
    dt_ignore[unmatched] = (pred_areas[unmatched] < area_min) | (pred_areas[unmatched] > area_max)

    return matched_pred, dt_ignore, gt_ignore


def _tv_similarity(probs, gt):
    """1 - total variation distance between two batches of distributions (N, C)."""
    return 1.0 - 0.5 * torch.sum(torch.abs(probs - gt), dim=1)


def _compute_ausrc(scores, similarities, bbox_tp, total_gt, return_curve=False):
    """Area Under the Similarity-Recall Curve: sorts detections by `scores`
    descending, tracks cumulative recall (cumulative TP / total_gt) and
    cumulative mean similarity, interpolates the latter at 101 recall levels
    (mirroring COCOeval's own 101-point precision interpolation for AP) and
    averages -- the soft-label analogue of AP's precision-recall integral, with
    "is this box's predicted distribution close to the GT's" (similarity)
    standing in for "is this box's predicted class exactly right" (precision).

    If `return_curve=True`, also returns the raw 101-point interpolated-
    similarity array (same one the mean is taken over) -- the actual
    similarity-vs-recall curve at whatever single IoU threshold this call was
    made at, category-agnostic by construction (AS never splits by class to
    begin with, unlike AP)."""
    order = torch.argsort(scores, descending=True)
    sims = similarities[order]
    tp = bbox_tp[order].float()

    cum_tp = torch.cumsum(tp, dim=0)
    recall = cum_tp / total_gt
    cum_mean_sim = torch.cumsum(sims, dim=0) / torch.arange(1, len(sims) + 1, device=sims.device)

    recall_levels = torch.linspace(0, 1, 101, device=sims.device)
    interp_sim = torch.zeros_like(recall_levels)
    for i, r in enumerate(recall_levels):
        mask = recall >= r
        interp_sim[i] = torch.max(cum_mean_sim[mask]) if mask.any() else 0.0

    if return_curve:
        return interp_sim.mean().item(), interp_sim.cpu().numpy()
    return interp_sim.mean().item()


def compute_similarity(process_data, soft_gt_path="../data/soft_datasets/soft_COCO_2017_val.json",
                        iou_thresholds=None, area_ranges=None, device=None, return_curve=False,
                        pr_curve_iou=None):
    """Computes "AS" (Average Similarity), the soft-label analogue of AP: instead of
    precision (is the detected class exactly right?), it measures 1 - TVD between
    the detection's predicted distribution (`p_hat`) and the GT's full 81-dim
    soft_label -- IoU-matched box-for-box, category-agnostically (soft labels
    already carry the class uncertainty, so there's no discrete GT category to
    match on) -- then averaged over recall levels and the 10 COCO IoU thresholds
    (0.5:0.95) exactly like AP, with the same small/medium/large object-size
    breakdown. Ranked by `score_est[:, 1]` (p(object present)), same as
    compute_map, so AS and AP are directly comparable at the same operating point.

    Unlike compute_map, the ground truth here is *not* thresholded to hard
    labels: every soft-GT object's full soft_label distribution is used directly,
    since this compares full distributions rather than discrete classes.

    Also computes "AS_strict" (+ its small/medium/large breakdown), a stricter
    variant that aligns AS with AP's own treatment of false positives: standard
    AS's `_tv_similarity` gives an unmatched/FP detection real, continuous credit
    whenever its `p_hat` happens to lean toward background -- a well-calibrated
    "this probably isn't an object" is rewarded even though AP's precision gives
    an FP zero credit no matter how it was scored. AS_strict keeps the exact same
    ranking/recall bookkeeping (FPs still occupy a rank in the sorted-by-score
    curve) but zeroes an FP's similarity contribution outright, so it degenerates
    to exactly AP if every `p_hat` were a hard one-hot. Both variants share the
    same per-image matching/`gt_full` computation, so this costs one extra
    `_compute_ausrc` call per (area, IoU threshold), not a second pass.

    Returns {"AS": ..., "AS_small": ..., "AS_medium": ..., "AS_large": ...,
    "AS_strict": ..., "AS_strict_small": ..., "AS_strict_medium": ...,
    "AS_strict_large": ...}.

    If `return_curve=True`, also adds "AS_curve" and "AS_strict_curve":
    {iou_thresh: value} dicts for area='all' at each of `iou_thresholds`' 10
    points -- the same per-threshold values "AS"/"AS_strict" already average
    over, exposed directly rather than discarded after averaging.

    If `pr_curve_iou` is set (e.g. 0.5), also adds "AS_pr_curve" and
    "AS_strict_pr_curve": {"recall": [101 values, 0..1], "similarity": [101
    values]} -- the actual similarity-vs-recall curve at that single IoU
    threshold, area='all' (already category-agnostic by construction, unlike
    AP's curve which has to be averaged over classes -- see compute_map)."""
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.5, 0.95, 10)
    if area_ranges is None:
        area_ranges = {"all": (0.0, 1e10), "small": (0.0, 32.0 ** 2),
                        "medium": (32.0 ** 2, 96.0 ** 2), "large": (96.0 ** 2, 1e10)}
    if device is None:
        # CPU by default: the per-image loop below does many tiny ops with CPU<->GPU
        # sync points, so kernel-launch/sync overhead would make GPU slower here.
        device = "cpu"

    gt_by_image = _load_soft_gt_by_image(soft_gt_path)

    # ---- per-image tensors, built once and shared across every (area_range, iou_thresh) combination ----
    image_data = {}
    for image_id, rows in process_data.groupby("image_id"):
        gt = gt_by_image.get(image_id)

        pred_boxes = torch.tensor(np.vstack(rows["gdino_bbox"].to_numpy()), dtype=torch.float32, device=device)
        pred_areas = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1])
        pred_scores = torch.tensor(np.vstack(rows["score_est"].to_numpy())[:, 1], dtype=torch.float32, device=device)
        probs = torch.tensor(np.vstack(rows["p_hat"].to_numpy()), dtype=torch.float32, device=device)

        if gt is not None and len(gt["boxes"]) > 0:
            gt_boxes = torch.tensor(gt["boxes"], dtype=torch.float32, device=device)
            gt_soft_labels = torch.tensor(gt["soft_labels"], dtype=torch.float32, device=device)
            gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (gt_boxes[:, 3] - gt_boxes[:, 1])
        else:
            gt_boxes = torch.empty((0, 4), dtype=torch.float32, device=device)
            gt_soft_labels = torch.empty((0, probs.shape[1]), dtype=torch.float32, device=device)
            gt_areas = torch.empty((0,), dtype=torch.float32, device=device)

        image_data[image_id] = dict(
            pred_scores=pred_scores, probs=probs, pred_areas=pred_areas,
            gt_soft_labels=gt_soft_labels, gt_areas=gt_areas,
            iou_matrix=_iou_matrix(pred_boxes, gt_boxes),
        )

    as_stats = {}
    as_strict_stats = {}
    all_area_curve, all_area_curve_strict = {}, {}
    pr_curve, pr_curve_strict = {}, {}
    recall_levels_np = np.linspace(0, 1, 101)
    for area_name, area_range in area_ranges.items():
        per_iou = []
        per_iou_strict = []
        for iou_thresh in iou_thresholds:
            want_pr_curve = area_name == "all" and pr_curve_iou is not None and np.isclose(iou_thresh, pr_curve_iou)
            all_scores, all_sims, all_sims_strict, all_tp = [], [], [], []
            total_gt = 0
            for data in image_data.values():
                matched_pred, dt_ignore, gt_ignore = _coco_style_matching(
                    data["iou_matrix"], data["pred_scores"], data["pred_areas"], data["gt_areas"],
                    iou_thresh, area_range,
                )
                gt_full = torch.zeros_like(data["probs"])
                gt_full[:, -1] = 1.0   # background one-hot for unmatched predictions
                matched_mask = matched_pred >= 0
                if matched_mask.any():
                    gt_full[matched_mask] = data["gt_soft_labels"][matched_pred[matched_mask]]

                valid = ~dt_ignore
                if valid.any():
                    sims = _tv_similarity(data["probs"][valid], gt_full[valid])
                    matched_valid = matched_mask[valid]
                    sims_strict = sims.clone()
                    sims_strict[~matched_valid] = 0.0   # FPs: zero credit, like AP's precision

                    all_scores.append(data["pred_scores"][valid])
                    all_sims.append(sims)
                    all_sims_strict.append(sims_strict)
                    all_tp.append(matched_valid)
                total_gt += int((~gt_ignore).sum().item())

            if total_gt == 0 or not all_scores:
                per_iou.append(float("nan"))
                per_iou_strict.append(float("nan"))
                continue
            scores_cat, tp_cat = torch.cat(all_scores), torch.cat(all_tp)
            if want_pr_curve:
                as_mean, as_curve_vals = _compute_ausrc(scores_cat, torch.cat(all_sims), tp_cat, total_gt,
                                                          return_curve=True)
                as_strict_mean, as_strict_curve_vals = _compute_ausrc(scores_cat, torch.cat(all_sims_strict),
                                                                       tp_cat, total_gt, return_curve=True)
                pr_curve = {"recall": recall_levels_np.tolist(), "similarity": as_curve_vals.tolist()}
                pr_curve_strict = {"recall": recall_levels_np.tolist(), "similarity": as_strict_curve_vals.tolist()}
                per_iou.append(as_mean)
                per_iou_strict.append(as_strict_mean)
            else:
                per_iou.append(_compute_ausrc(scores_cat, torch.cat(all_sims), tp_cat, total_gt))
                per_iou_strict.append(_compute_ausrc(scores_cat, torch.cat(all_sims_strict), tp_cat, total_gt))
        as_stats[area_name] = float(np.nanmean(per_iou))
        as_strict_stats[area_name] = float(np.nanmean(per_iou_strict))
        if area_name == "all":
            all_area_curve = {round(float(t), 2): v for t, v in zip(iou_thresholds, per_iou)}
            all_area_curve_strict = {round(float(t), 2): v for t, v in zip(iou_thresholds, per_iou_strict)}

    result = {"AS": as_stats["all"], "AS_small": as_stats["small"],
            "AS_medium": as_stats["medium"], "AS_large": as_stats["large"],
            "AS_strict": as_strict_stats["all"], "AS_strict_small": as_strict_stats["small"],
            "AS_strict_medium": as_strict_stats["medium"], "AS_strict_large": as_strict_stats["large"]}

    if return_curve:
        result["AS_curve"] = all_area_curve
        result["AS_strict_curve"] = all_area_curve_strict

    if pr_curve_iou is not None:
        result["AS_pr_curve"] = pr_curve
        result["AS_strict_pr_curve"] = pr_curve_strict

    return result


def evaluate_result(process_data, dataset_name=None, only_objects=None, data_space=None,
                     label_column=None, prob_label_column=None, n_data=None, k=None, alpha=None,
                     n_samples=None, config=None, elapsed_time=None, time_NN=None, time_graph=None,
                     time_setup=None, time_solver=None, time_spread=None, compute_ap=False, compute_as=False,
                     soft_gt_path="../data/soft_datasets/soft_COCO_2017_val.json",
                     soft_label_thresh=0.5, ap_verbose=False):
    """Computes accuracy/TVD/RMSE/MAE metrics from `process_data` -- a dataframe that
    already has 'prediction', 'p_hat', 'score_est', 'prediction_mod' and 'p_hat_mod'
    columns set, as returned by prob_label_spreading_object,
    label_spreading_object_conventional, knn_regression_object or gkr_object (all in
    src.py) -- and assembles the single-row results DataFrame. Decoupled from those
    functions so additional/alternative evaluation procedures can be run over the
    same process_data without re-running the (expensive) spreading/regression step.

    Every descriptive/bookkeeping parameter (dataset_name, only_objects, ..., the
    timing breakdown) is optional and falls back to `process_data.attrs["run_meta"]`,
    which the functions above populate automatically -- pass a parameter explicitly
    only to override it (e.g. after the dataframe was copied/reloaded and lost its
    .attrs, since pandas does not guarantee attrs survive every operation).

    compute_ap=True additionally runs compute_map(process_data, soft_gt_path,
    soft_label_thresh, verbose=ap_verbose) (COCO mAP/AR via pycocotools against the
    soft GT thresholded into hard labels, see compute_map's docstring) twice --
    once ranking by the default TP/FP-diffusion objectness (`score_est[:, 1]`,
    merged as AP/AP50/.../AR_large), and once ranking by the
    soft-label diffusion's own max non-background class probability
    (`p_hat[:, :-1].max(axis=1)`, merged with an "_sl" suffix: AP_sl, AP50_sl,
    ..., AR_large_sl) -- both cost the same COCOeval call, so it's cheap to
    always compare a TP/FP-based vs. a purely soft-label-based re-ranking.

    compute_as=True additionally runs compute_similarity(process_data, soft_gt_path)
    (Average Similarity, the soft-label analogue of AP, see its docstring) and
    merges AS/AS_small/AS_medium/AS_large and the stricter AS_strict/AS_strict_small/
    AS_strict_medium/AS_strict_large (FPs score 0, like AP, instead of getting
    credit for a background-leaning p_hat -- see compute_similarity's docstring)
    into the result row."""
    meta = dict(process_data.attrs.get("run_meta", {}))
    overrides = dict(dataset_name=dataset_name, only_objects=only_objects, data_space=data_space,
                      label_column=label_column, prob_label_column=prob_label_column, n_data=n_data,
                      k=k, alpha=alpha, n_samples=n_samples, config=config, elapsed_time=elapsed_time,
                      time_NN=time_NN, time_graph=time_graph, time_setup=time_setup,
                      time_solver=time_solver, time_spread=time_spread)
    meta.update({key: val for key, val in overrides.items() if val is not None})

    required = ("dataset_name", "only_objects", "data_space", "label_column", "prob_label_column")
    missing = [key for key in required if meta.get(key) is None]
    if missing:
        raise ValueError(f"evaluate_result: missing required parameter(s) {missing} -- pass them "
                          f"explicitly, or call evaluate_result on process_data as returned directly "
                          f"from a spreading/regression function (its run_meta carries them).")

    dataset = process_data
    dataset_name, only_objects, data_space = meta["dataset_name"], meta["only_objects"], meta["data_space"]
    label_column, prob_label_column = meta["label_column"], meta["prob_label_column"]
    n_data = meta.get("n_data") if meta.get("n_data") is not None else len(dataset)
    k, alpha, n_samples, config = meta.get("k"), meta.get("alpha"), meta.get("n_samples"), meta.get("config")
    bandwidth_scale = meta.get("bandwidth_scale", 1.0)
    elapsed_time = meta.get("elapsed_time", np.nan)
    time_NN, time_graph = meta.get("time_NN", np.nan), meta.get("time_graph", np.nan)
    time_setup, time_solver, time_spread = meta.get("time_setup", np.nan), meta.get("time_solver", np.nan), meta.get("time_spread", np.nan)

    # restrict to matched predictions (gdino_match == 1)
    matches = dataset[dataset["gdino_match"] == 1]

    y_true = dataset[label_column].astype(int)
    y_pred = dataset["prediction"].astype(int)
    y_pred_mod = dataset["prediction_mod"].astype(int)

    y_true_match = matches[label_column].astype(int)
    y_pred_match = matches["prediction"].astype(int)
    y_pred_match_mod = matches["prediction_mod"].astype(int)


    acc = (y_true == y_pred).mean()
    acc_mod = (y_true == y_pred_mod).mean()
    acc_match = (y_true_match == y_pred_match).mean()
    acc_match_mod = (y_true_match == y_pred_match_mod).mean()

    p = np.vstack(dataset[prob_label_column].values)
    p_hat = np.vstack(dataset["p_hat"].values)
    p_hat_match = np.vstack(matches["p_hat"].values)
    p_hat_mod = np.vstack(dataset["p_hat_mod"].values)


    # matches and mod for other eval metrics
    p_match = np.vstack(matches[prob_label_column].values)
    p_hat_match_mod = np.vstack(matches["p_hat_mod"].values)

    # mean objectness estimate over all samples (reported as Obj_p_* when only_objects)
    obj_scores = np.vstack(dataset["score_est"].to_numpy())
    global_obj_scores = obj_scores.mean(axis=0)
    obj_scores_match = np.vstack(matches["score_est"].to_numpy())
    global_obj_scores_match = obj_scores_match.mean(axis=0)


    ### total variation distance
    tvd = 0.5 * np.abs(p_hat - p).sum(axis=1).mean()
    tvd_macth = 0.5 * np.abs(p_hat_match - p_match).sum(axis=1).mean()
    # per sample
    dataset["tvd_sample"] = 0.5 * np.abs(p_hat - p).sum(axis=1)
    tvd_mod = 0.5 * np.abs(p_hat_mod - p).sum(axis=1).mean()
    tvd_match_mod = 0.5 * np.abs(p_hat_match_mod - p_match).sum(axis=1).mean()
    # per sample tvd
    dataset["tvd_sample_mod"] = 0.5 * np.abs(p_hat_mod - p).sum(axis=1)

    rmse = sklearn.metrics.root_mean_squared_error(p_hat, p)
    rmse_match = sklearn.metrics.root_mean_squared_error(p_hat_match, p_match)

    rmse_mod = sklearn.metrics.root_mean_squared_error(p_hat_mod, p)
    rmse_match_mod = sklearn.metrics.root_mean_squared_error(p_hat_match_mod, p_match)

    mae = sklearn.metrics.mean_absolute_error(p_hat, p)
    mae_match = sklearn.metrics.mean_absolute_error(p_hat_match, p_match)

    mae_mod = sklearn.metrics.mean_absolute_error(p_hat_mod, p)
    mae_match_mod = sklearn.metrics.mean_absolute_error(p_hat_match_mod, p_match)

    data_dim = len(dataset[data_space].iloc[0])

    if data_space == "feature":
        dim_reduction_technique = "NONE"
    elif len(data_space.split("_")[:-1]) > 1: # combination of e.g. CLIP_UMAP
        dim_reduction_technique = "+".join(data_space.split("_")[:-1])
    else:
        dim_reduction_technique = data_space.split("_")[0] # just one technique e.g. CLIP

    if not only_objects:
        result = pd.DataFrame({"dataset": dataset_name, "n_data": n_data, "dim_reduction_technique": dim_reduction_technique, "dimension": data_dim, "only_objects": only_objects , "prob_labels": prob_label_column,\
         'alpha':alpha,'n_samples': n_samples, "k": k, "bandwidth_scale": bandwidth_scale,
         'RMSE':rmse,
         'RMSE_mod': rmse_mod, 'RMSE_match':rmse_match, 'RMSE_match_mod':rmse_match_mod,
         'Acc':acc, 'Acc_match':acc_match,
         'Acc_mod':acc_mod,'Acc_match_mod':acc_match_mod,
         "TVD": tvd, "TVD_match": tvd_macth,
         "TVD_mod": tvd_mod, "TVD_match_mod": tvd_match_mod,
         "MAE": mae, "MAE_match": mae_match,
         "MAE_mod": mae_mod,"MAE_match_mod": mae_match_mod,
         "runtime": elapsed_time, "time_NN": time_NN, "time_graph": time_graph, "time_setup": time_setup,  "time_solver": time_solver, "time_spread": time_spread,
         "config": config}, index=[0])

    else:
        result = pd.DataFrame({"dataset": dataset_name, "n_data": n_data, "dim_reduction_technique": dim_reduction_technique, "dimension": data_dim, "only_objects": only_objects , "prob_labels": prob_label_column,\
            'alpha':alpha,'n_samples': n_samples, "k": k, "bandwidth_scale": bandwidth_scale,
            'RMSE':rmse, "RMSE_match": rmse_match,
            'RMSE_mod': rmse_mod, 'RMSE_match_mod':rmse_match_mod,
            'Acc':acc, "Acc_match": acc_match,
            'Acc_mod':acc_mod, 'Acc_match_mod':acc_match_mod,
            "TVD": tvd, "TVD_match": tvd_macth,
            "TVD_mod": tvd_mod, "TVD_match_mod": tvd_match_mod,
            "MAE": mae, "MAE_match": mae_match,
            "MAE_mod": mae_mod, "MAE_match_mod": mae_match_mod,
            "Obj_p_bg": float(global_obj_scores[0]), "Obj_p_obj": float(global_obj_scores[1]), "Obj_p_bg_match": float(global_obj_scores_match[0]),"Obj_p_obj_match": float(global_obj_scores_match[1]),
            "runtime": elapsed_time, "time_NN": time_NN, "time_graph": time_graph, "time_setup": time_setup,  "time_solver": time_solver, "time_spread": time_spread,
            "config": config}, index=[0])

    if compute_ap:
        ap_stats = compute_map(dataset, soft_gt_path=soft_gt_path, soft_label_thresh=soft_label_thresh, verbose=ap_verbose)
        for stat_name, value in ap_stats.items():
            result[stat_name] = value

        # same re-ranking, but scored by the soft-label diffusion alone (max
        # non-background class probability) instead of the TP/FP-diffusion
        # objectness score -- same COCOeval cost, so worth always comparing.
        p_hat_scores = p_hat[:, :-1].max(axis=1)
        ap_stats_sl = compute_map(dataset, soft_gt_path=soft_gt_path, soft_label_thresh=soft_label_thresh,
                                   verbose=ap_verbose, scores=p_hat_scores)
        for stat_name, value in ap_stats_sl.items():
            result[f"{stat_name}_sl"] = value

    if compute_as:
        as_stats = compute_similarity(dataset, soft_gt_path=soft_gt_path)
        for stat_name, value in as_stats.items():
            result[stat_name] = value

    return result
