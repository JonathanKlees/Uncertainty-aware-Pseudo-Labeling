"""Active-learning query experiment for PLS (results visualized in
analysis/active_querying.ipynb):

  1. Query strategies: random, entropy (classification uncertainty), trials (how
     little propagated evidence a node has received) and "mixture", blending
     entropy and trials into one acquisition score.
  2. One-annotation-at-a-time querying: after every single newly-revealed label,
     the PLS system is re-solved and the acquisition function re-evaluated
     against the updated state.
  3. Evaluation (compute_ap + compute_as, ~2 min/call via pycocotools -- two orders
     of magnitude more expensive than a single PLS solve, ~1s/call) only runs
     every EVAL_EVERY=100 newly-labeled instances, not every query -- otherwise
     evaluation alone would dominate total runtime even more than it already does.
  4. Two strategies based on the heat-kernel-weighted Wilson interval width,
     K = (I - alpha S)^-1 and w_i the class-summed 95% Wilson width at node i:
     "wilson_kernel" queries argmax_j sum_i K_ij w_i (see query_wilson_kernel),
     "wilson_kernel_norm" the kernel-weighted mean sum_i K_ij w_i / sum_i K_ij
     (see query_wilson_kernel_norm; shown as "Graph-aware Uncertainty" in
     analysis/active_querying.ipynb). Each costs one extra linear solve per query.
     At low budgets the unnormalized score reduces to graph centrality K 1, which
     is why only the normalized variant is reported.

Because querying is continuous (not re-run per fixed target budget), one run up
to TARGET_BUDGET passes through every smaller budget along the way -- e.g. the
n_labeled=900 checkpoint of a run targeting 10% is the ~1% result, no separate
re-run needed.

Extending TARGET_BUDGET upward (e.g. 5% -> 10%) resumes each strategy from its
last completed checkpoint via the deterministic-replay mechanism in
run_active_learning_single (replay_to): re-walks the already-known part of the
trajectory solve-only (cheap, ~1s/step) without re-evaluating (expensive,
~2 min/checkpoint), then continues querying/evaluating for real past the old
ceiling. A strategy whose last checkpoint already reached the (new) target is
skipped entirely.

Long-running: one strategy up to the 10% budget (8,599 labels, 86 evaluated
checkpoints at ~2 min each) takes about 4.5-5h on COCO val, so all six
strategies from scratch take on the order of 30h -- run via tmux, not
interactively. Results are appended to CSV after every single checkpoint (not
just per-strategy) given the multi-hour runtime, and each strategy's full query order
is stored alongside (query_history_path) for the spread/UMAP analysis, so it never
needs a separate replay (reconstruct_query_history.py, for older runs).
"""
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import pyamgx
import scipy.sparse

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

from src import spreading_conventional, set_graph, _apply_object_conditioning, _ensure_amgx_initialized
from evaluation import evaluate_result

pd.set_option("display.max_columns", None)

DATASET_PATH = "../data/matched_dataset/coco_val_matched.parquet"
SOFT_GT_PATH = "../data/soft_datasets/soft_COCO_2017_val.json"
AMGX_CONFIG  = "./configs/FGMRES_AGGREGATION.json"
GRID_CSV     = "../analysis/experiments/coco_val_baseline_comp_simple/grid_search/coco_val_baseline_comp_simple_grid_search_results.csv"
RESULTS_CSV  = "../analysis/experiments/active_learning_single_query_results.csv"


def query_history_path(k, alpha):
    """Per-strategy query order for the spread/UMAP analysis in analysis/active_querying.ipynb."""
    return f"../analysis/experiments/active_learning_prototype_query_history_k{k}_alpha{alpha}.pkl"


def save_query_history(path, strategy, order, k, alpha, seed_indices, n_data):
    """Stores one strategy's query order -- [("seed", [...]), ("query", idx), ...] -- in the
    shared history pickle, keeping all other strategies' entries; written atomically."""
    qh = dict(history={}, K=k, ALPHA=alpha)
    if os.path.exists(path):
        with open(path, "rb") as f:
            qh = pickle.load(f)
        assert (qh["K"], qh["ALPHA"]) == (k, alpha), f"{path} was built with other (k, alpha)"
    qh["history"][strategy] = order
    qh.update(seed_indices=seed_indices, n_data=n_data)
    with open(path + ".tmp", "wb") as f:
        pickle.dump(qh, f)
    os.replace(path + ".tmp", path)

SCORE_THR   = 0.2
DATA_SPACE  = "gdino_embedding"
LABEL_COL   = "label"
PROB_COL    = "soft_label"
SCORE_COL   = "is_object"
SOFT_LABEL_THRESH = 0.5

TARGET_BUDGET = 0.10     # continuous run's ceiling -- passes through 1%/2%/.../5%/... along the way
# (k, alpha) for the whole run: FIXED_HP if set, otherwise the AP-optimal PLS grid point at
# HP_BUDGET. Replay requires the exact (k, alpha) that produced the already-evaluated part
# of a trajectory, so keep both fixed when raising TARGET_BUDGET.
HP_BUDGET     = 0.05
FIXED_HP      = (20, 0.9)
EVAL_EVERY    = 100      # evaluate (expensive) only every this-many newly-labeled instances
SEED_N        = 100      # initial random seed batch, shared across strategies (== EVAL_EVERY)
STRATEGIES    = ["random", "entropy", "trials", "mixture", "wilson_kernel", "wilson_kernel_norm"]
MIXTURE_WEIGHT = 0.5     # 0.5 = equal blend of entropy and low-trials; 1.0 -> pure entropy, 0.0 -> pure trials
WILSON_Z      = 1.96     # 95% Wilson score interval, for the wilson_kernel strategy
SEED          = 42


# ── 1. Query strategies ──────────────────────────────────────────────────────
def query_random(candidate_idx, n_query, rng):
    n_query = min(n_query, len(candidate_idx))
    return rng.choice(candidate_idx, size=n_query, replace=False)


def query_entropy(p_hat, candidate_idx, n_query):
    """Highest Shannon entropy of p_hat among candidates -- most classification uncertainty."""
    n_query = min(n_query, len(candidate_idx))
    p = p_hat[candidate_idx]
    entropy = -(p * np.log(p + 1e-12)).sum(axis=1)
    top = np.argsort(-entropy)[:n_query]
    return candidate_idx[top]


def query_low_trials(trials, candidate_idx, n_query):
    """Least total propagated mass among candidates -- least evidence has reached this node,
    regardless of what its current (possibly confident-but-uninformed) p_hat says."""
    n_query = min(n_query, len(candidate_idx))
    t = trials[candidate_idx]
    top = np.argsort(t)[:n_query]
    return candidate_idx[top]


def _minmax(x):
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


def query_mixture(p_hat, trials, candidate_idx, n_query, weight=MIXTURE_WEIGHT):
    """Weighted blend of entropy (uncertainty) and inverse-trials (evidence starvation),
    each min-max normalized over the candidate pool before combining -- puts both criteria
    on a common [0,1] scale so neither dominates purely from having a wider numeric range.
    weight=1.0 reduces to pure query_entropy, weight=0.0 to pure query_low_trials."""
    n_query = min(n_query, len(candidate_idx))
    p = p_hat[candidate_idx]
    entropy = -(p * np.log(p + 1e-12)).sum(axis=1)
    t = trials[candidate_idx]

    score = weight * _minmax(entropy) + (1 - weight) * (1 - _minmax(t))
    top = np.argsort(-score)[:n_query]
    return candidate_idx[top]


def wilson_width(p_hat, trials, z=WILSON_Z):
    """Per-node Wilson score interval width, summed over all classes: p_hat as the
    proportion and the propagated mass `trials` as the (effective, non-integer) number of
    observations. Written as 2z*sqrt(n*p*(1-p) + z^2/4) / (n + z^2), which equals the usual
    2z/(1+z^2/n) * sqrt(p(1-p)/n + z^2/(4n^2)) but stays finite at n=0, where each class's
    width is 1 (no evidence at all -> maximally wide)."""
    n = trials[:, None]
    per_class = 2 * z * np.sqrt(n * p_hat * (1 - p_hat) + z ** 2 / 4) / (n + z ** 2)
    return per_class.sum(axis=1)


def query_wilson_kernel(p_hat, trials, candidate_idx, n_query, S, alpha, config):
    """Candidate maximizing the heat-kernel-weighted Wilson width s_j = sum_i K_ij w_i,
    K = (I - alpha S)^-1 -- the node whose label would propagate most strongly into
    currently uncertain nodes (its own width included, via K_jj). S is symmetric
    (D^-1/2 W D^-1/2), so K is too and s = K w: a single extra linear solve per query;
    K is never formed."""
    n_query = min(n_query, len(candidate_idx))
    s = kernel_solve(S, alpha, wilson_width(p_hat, trials), config)
    top = np.argsort(-s[candidate_idx])[:n_query]
    return candidate_idx[top]


def query_wilson_kernel_norm(p_hat, trials, candidate_idx, n_query, S, alpha, config, kernel_row_sum):
    """query_wilson_kernel's score divided by the kernel's row sums K 1, i.e. the
    kernel-weighted MEAN Wilson width around each candidate. At low budgets trials << z^2
    almost everywhere, so the widths are nearly constant and the unnormalized K w ranks
    nodes almost exactly like K 1 (graph centrality, Spearman 0.999 at 1,100 labels);
    normalizing removes that and leaves where the surrounding evidence is weakest.
    kernel_row_sum = K 1 does not change during a run, so it is solved once up front."""
    n_query = min(n_query, len(candidate_idx))
    s = kernel_solve(S, alpha, wilson_width(p_hat, trials), config) / kernel_row_sum
    top = np.argsort(-s[candidate_idx])[:n_query]
    return candidate_idx[top]


# ── 2. Single-round PLS solve + (expensive) evaluation ──────────────────────
# solve_conventional (src.py) draws its one-hot class samples via the global, unseeded
# np.random state, so two solves only agree (up to ~1e-7 AMGX noise) if np.random.seed is
# set to the same value right before. Replay reconstructs a trajectory in a new process,
# so every solve is seeded by (strategy, n_labeled) via a fixed per-strategy offset (not
# Python's hash(), which is randomized per process).
STRATEGY_SEED_OFFSET = {"random": 0, "entropy": 1_000_000, "trials": 2_000_000, "mixture": 3_000_000,
                        "wilson_kernel": 4_000_000, "wilson_kernel_norm": 5_000_000}


def solve_round(S, alpha, dataset_in, classes_in, indices, config, seed=None):
    if seed is not None:
        np.random.seed(seed)
    F, F_object, _, _, _ = spreading_conventional(
        alpha, S, dataset_in, PROB_COL, SCORE_COL, classes_in, config, indices,
    )
    row_sum = F.sum(axis=1)
    uniform = np.full(len(classes_in), 1.0 / len(classes_in))
    p_hat = np.divide(F, row_sum[:, None], out=np.tile(uniform, (len(dataset_in), 1)), where=row_sum[:, None] > 0)

    obj_row_sum = F_object.sum(axis=1)
    score_est = np.divide(F_object, obj_row_sum[:, None], out=np.full((len(dataset_in), 2), 0.5),
                           where=obj_row_sum[:, None] > 0)
    return p_hat, score_est, row_sum


def kernel_solve(S, alpha, rhs, config):
    """x = (I - alpha S)^-1 rhs, one AMGX solve with its own setup (~10ms setup + ~5ms
    solve at COCO val scale, negligible next to solve_round). Kept separate from
    src.solve_conventional so the existing strategies' code path stays untouched."""
    _ensure_amgx_initialized()
    pyamgx.register_print_callback(lambda *args, **kwargs: None)
    cfg = pyamgx.Config().create_from_file(config)
    rsc = pyamgx.Resources().create_simple(cfg)
    A = pyamgx.Matrix().create(rsc)
    b = pyamgx.Vector().create(rsc)
    x = pyamgx.Vector().create(rsc)
    solver = pyamgx.Solver().create(rsc, cfg)
    try:   # free GPU objects even if setup/solve raises, as in solve_conventional
        A.upload_CSR(scipy.sparse.csr_matrix(scipy.sparse.eye(S.shape[0]) - alpha * S))
        solver.setup(A)
        sol = np.zeros(S.shape[0], dtype=np.float64)
        b.upload(np.ascontiguousarray(rhs, dtype=np.float64))
        x.upload(sol)
        solver.solve(b, x, zero_initial_guess=True)
        x.download(sol)
    finally:
        for obj in (A, x, b, solver, rsc, cfg):
            obj.destroy()
    return sol


def evaluate_checkpoint(dataset_in, p_hat, score_est, run_tag, k, alpha, n_samples):
    ds = dataset_in.copy()
    ds["p_hat"] = list(p_hat)
    ds["prediction"] = p_hat.argmax(axis=1)
    ds["score_est"] = list(score_est)
    _apply_object_conditioning(ds)
    ds.attrs["run_meta"] = dict(
        dataset_name=run_tag, only_objects=False, data_space=DATA_SPACE,
        label_column=LABEL_COL, prob_label_column=PROB_COL, n_data=len(ds), k=k, alpha=alpha,
        n_samples=n_samples, config=None, elapsed_time=0.0, time_NN=0.0, time_graph=0.0,
        time_setup=0.0, time_solver=0.0, time_spread=0.0,
    )
    return evaluate_result(ds, compute_ap=True, compute_as=True,
                            soft_gt_path=SOFT_GT_PATH, soft_label_thresh=SOFT_LABEL_THRESH).iloc[0]


# ── 3. True one-at-a-time active-learning loop ───────────────────────────────
def run_active_learning_single(dataset_in, S, classes_in, k, alpha, total_budget_n,
                                strategy, seed_indices, rng, eval_every=EVAL_EVERY, config=AMGX_CONFIG,
                                on_checkpoint=None, replay_to=0, on_query=None):
    """Queries ONE instance at a time (re-solving PLS after every single reveal, so the
    acquisition function always sees the current, fully up-to-date state) but only runs
    the expensive evaluate_checkpoint every `eval_every` newly-labeled instances. Only
    checkpointed (evaluated) states are kept in memory / returned -- intermediate
    un-evaluated p_hat/trials snapshots are used solely to drive the next single query
    and then discarded, since storing a full (n_data, n_classes) array at every one of
    up to ~8,500 individual queries would be far too much memory.

    on_checkpoint(n_labeled, p_hat, score_est, trials), if given, is called immediately
    at every checkpoint (not just returned at the end) so the caller can evaluate + save
    incrementally -- essential here since a single strategy run takes ~4.5-5h.

    on_query(new_idx), if given, is called with every newly-queried index, in order --
    main() uses it to record the full query sequence (see query_history_path).

    replay_to>0 resumes a previous lower-budget run of the SAME strategy: given the same
    seed_indices/rng AND solve_round's per-step seed pinned to (strategy, n_labeled) (see
    STRATEGY_SEED_OFFSET above), the whole trajectory is deterministic, so re-walking from
    n_labeled=len(seed_indices) up to replay_to reproduces the exact same query sequence
    and PLS states already evaluated in a prior run -- so on_checkpoint is suppressed
    until n_labeled exceeds replay_to, meaning the (expensive, ~2 min each) evaluations
    for those already-known checkpoints are never redone, only the (cheap, ~1s) solves
    that must still run to reconstruct the state needed to keep querying forward from
    there."""
    n = len(dataset_in)
    labeled_mask = np.zeros(n, dtype=bool)
    labeled_mask[seed_indices] = True
    indices = np.where(labeled_mask)[0]

    seed_offset = STRATEGY_SEED_OFFSET[strategy]
    n_labeled = int(labeled_mask.sum())
    p_hat, score_est, trials = solve_round(S, alpha, dataset_in, classes_in, indices, config,
                                            seed=seed_offset + n_labeled)
    if on_checkpoint is not None and n_labeled > replay_to:
        on_checkpoint(n_labeled, p_hat, score_est, trials)

    next_eval_at = ((n_labeled // eval_every) + 1) * eval_every
    kernel_row_sum = kernel_solve(S, alpha, np.ones(n), config) if strategy == "wilson_kernel_norm" else None

    while n_labeled < total_budget_n:
        candidate_idx = np.where(~labeled_mask)[0]
        if len(candidate_idx) == 0:
            break

        if strategy == "random":
            new_idx = query_random(candidate_idx, 1, rng)
        elif strategy == "entropy":
            new_idx = query_entropy(p_hat, candidate_idx, 1)
        elif strategy == "trials":
            new_idx = query_low_trials(trials, candidate_idx, 1)
        elif strategy == "mixture":
            new_idx = query_mixture(p_hat, trials, candidate_idx, 1)
        elif strategy == "wilson_kernel":
            new_idx = query_wilson_kernel(p_hat, trials, candidate_idx, 1, S, alpha, config)
        elif strategy == "wilson_kernel_norm":
            new_idx = query_wilson_kernel_norm(p_hat, trials, candidate_idx, 1, S, alpha, config, kernel_row_sum)
        else:
            raise ValueError(f"unknown strategy {strategy!r}")

        labeled_mask[new_idx] = True
        if on_query is not None:
            on_query(int(new_idx[0]))
        indices = np.where(labeled_mask)[0]
        n_labeled = int(labeled_mask.sum())
        p_hat, score_est, trials = solve_round(S, alpha, dataset_in, classes_in, indices, config,
                                                seed=seed_offset + n_labeled)

        if n_labeled >= next_eval_at or n_labeled == total_budget_n:
            if on_checkpoint is not None and n_labeled > replay_to:
                on_checkpoint(n_labeled, p_hat, score_est, trials)
            next_eval_at += eval_every


# ── 4. Setup ──────────────────────────────────────────────────────────────────
def main():
    print("Loading dataset...", flush=True)
    df_full = pd.read_parquet(DATASET_PATH)
    dataset = df_full[df_full["gdino_score"] >= SCORE_THR].copy().reset_index(drop=True)
    dataset["soft_label"] = list(np.array([v for v in dataset["soft_label"].values]))
    dataset["is_object"] = dataset["gdino_match"].astype(float)
    n_data = len(dataset)
    classes = sorted(dataset[LABEL_COL].dropna().unique())
    print(f"n_data={n_data:,}  classes={len(classes)}  match rate={dataset['gdino_match'].mean():.1%}", flush=True)

    if FIXED_HP is not None:
        k, alpha = FIXED_HP
        print(f"Fixed (k, alpha) for the whole continuous run: k={k}  alpha={alpha}", flush=True)
    else:
        grid = pd.read_csv(GRID_CSV)
        pls_grid = grid[grid["method"] == "PLS"]
        sub = pls_grid[np.isclose(pls_grid["budget"], HP_BUDGET)]
        best = sub.loc[sub["AP"].idxmax()]
        k, alpha = int(best["k"]), float(best["alpha"])
        print(f"Fixed (k, alpha) for the whole continuous run (AP-optimal at budget={HP_BUDGET:.0%}, "
              f"pinned there for replay consistency even though the target ceiling is {TARGET_BUDGET:.0%}): "
              f"k={k}  alpha={alpha}", flush=True)

    total_budget_n = max(1, int(n_data * TARGET_BUDGET))
    print(f"Target ceiling: n_labeled={total_budget_n:,} ({TARGET_BUDGET:.0%} of {n_data:,})", flush=True)

    t0 = time.time()
    S, _, time_NN, time_graph = set_graph(k, dataset, DATA_SPACE, n_data=n_data)
    print(f"Graph built in {time.time() - t0:.1f}s (NN={time_NN:.1f}s, graph={time_graph:.1f}s)", flush=True)

    # shared across strategies so every strategy starts from the identical seed set --
    # isolates the effect of the acquisition strategy itself, not the random seed draw
    seed_rng = np.random.default_rng(SEED)
    seed_indices = seed_rng.choice(n_data, size=SEED_N, replace=False)

    results_rows = []
    resume_from = {}   # strategy -> max n_labeled already checkpointed in a prior (lower-budget) run
    if os.path.exists(RESULTS_CSV):
        prior = pd.read_csv(RESULTS_CSV)
        # old and new checkpoints form one continuous run up to the current TARGET_BUDGET
        prior["target_budget"] = TARGET_BUDGET
        results_rows = prior.to_dict("records")
        for strategy, grp in prior.groupby("strategy"):
            resume_from[strategy] = int(grp["n_labeled"].max())
        print(f"Resuming from prior checkpoints: {resume_from}", flush=True)

    def save():
        pd.DataFrame(results_rows).to_csv(RESULTS_CSV, index=False)

    for strategy in STRATEGIES:
        replay_to = resume_from.get(strategy, 0)
        if replay_to >= total_budget_n:
            print(f"\n=== {strategy}: already reached n_labeled={replay_to:,} >= target "
                  f"{total_budget_n:,}, skipping ===", flush=True)
            continue

        if replay_to > 0:
            print(f"\n=== {strategy}: resuming from n_labeled={replay_to:,} (replaying solve-only, no "
                  f"re-evaluation, up to there) to new target n_labeled={total_budget_n:,} ===", flush=True)
        else:
            print(f"\n=== {strategy}: starting continuous single-instance run to n_labeled={total_budget_n:,} ===",
                  flush=True)
        t_strategy = time.time()
        rng = np.random.default_rng(SEED)
        n_checkpoints = [0]
        # full query order, recorded alongside the checkpoints so the spread/UMAP analysis
        # never needs a separate replay (a resumed run re-walks the earlier queries, so
        # this is complete again once the strategy finishes)
        query_order = [("seed", seed_indices.tolist())]

        def save_order(strategy=strategy, query_order=query_order):
            save_query_history(query_history_path(k, alpha), strategy, query_order, k, alpha,
                               seed_indices, n_data)

        def on_checkpoint(n_labeled, p_hat, score_est, trials, strategy=strategy, t_strategy=t_strategy):
            t0 = time.time()
            res = evaluate_checkpoint(dataset, p_hat, score_est, f"al_single_{strategy}_{n_labeled}",
                                       k, alpha, n_labeled)
            dt_eval = time.time() - t0
            # AP_sl: same COCOeval call, ranked by the max non-background class probability
            # (p_hat) instead of the objectness estimate (score_est) used by AP
            results_rows.append(dict(
                target_budget=TARGET_BUDGET, strategy=strategy, n_labeled=n_labeled,
                budget_frac=n_labeled / n_data, AP=res["AP"], AP_sl=res["AP_sl"],
                AS=res["AS"], AS_strict=res["AS_strict"],
                elapsed_s=time.time() - t_strategy,
            ))
            save()
            save_order()
            n_checkpoints[0] += 1
            print(f"  [{strategy}] n_labeled={n_labeled:5d}/{total_budget_n}  "
                  f"AP={res['AP']:.4f}  AP_sl={res['AP_sl']:.4f}  AS={res['AS']:.4f}  "
                  f"AS_strict={res['AS_strict']:.4f}  (eval took {dt_eval:.1f}s, "
                  f"{time.time() - t_strategy:.0f}s elapsed)", flush=True)

        run_active_learning_single(dataset, S, classes, k, alpha, total_budget_n, strategy,
                                    seed_indices, rng, eval_every=EVAL_EVERY, config=AMGX_CONFIG,
                                    on_checkpoint=on_checkpoint, replay_to=replay_to,
                                    on_query=lambda idx, query_order=query_order: query_order.append(("query", idx)))
        save_order()

        print(f"  {strategy} done in {time.time() - t_strategy:.0f}s ({n_checkpoints[0]} checkpoints)", flush=True)

    print(f"\nSaved -> {RESULTS_CSV}", flush=True)


if __name__ == "__main__":
    main()
