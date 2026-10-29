"""Reconstructs the per-step queried-index history of single-query AL strategies, for
the query-spread metrics and UMAP plots in analysis/active_querying.ipynb (Section 1).

active_querying.py's run_active_learning_single only reports eval checkpoints, not
which index got queried at each step. Its trajectory is deterministic, though:
solve_round's sampling seed is pinned to (strategy, n_labeled), and the shared seed
set and random-strategy rng are seeded with SEED. This script therefore re-runs
run_active_learning_single SOLVE-ONLY (no evaluation) up to TARGET_BUDGET and records
every newly-queried index through its on_query callback -- the same code path as the
real run, so it reproduces the same query sequence (up to AMGX floating-point noise,
which can flip exact near-ties between near-duplicate detections).

active_querying.py's main() also records the query order into the same pickle; this
script fills in strategies missing from it. Strategies already present in the history
pickle are kept as they are; only the requested ones that are missing get
reconstructed (~1-2s per query, i.e. a few hours per strategy at the 10% budget --
run via tmux). Run from this directory:
    python reconstruct_query_history.py --strategies wilson_kernel_norm
"""
import argparse
import os
import pickle
import shutil
import time

import numpy as np
import pandas as pd

import active_querying as aq


def main(strategies):
    k, alpha = aq.FIXED_HP
    OUT_PATH = aq.query_history_path(k, alpha)
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH, "rb") as f:
            qh = pickle.load(f)
        assert (qh["K"], qh["ALPHA"]) == (k, alpha), "history pickle was built with other (k, alpha)"
        shutil.copy2(OUT_PATH, OUT_PATH + ".bak")
    else:
        qh = dict(history={}, K=k, ALPHA=alpha)
    todo = [s for s in strategies if s not in qh["history"]]
    print(f"already in history: {list(qh['history'])}  ->  reconstructing: {todo}", flush=True)

    df_full = pd.read_parquet(aq.DATASET_PATH)
    dataset = df_full[df_full["gdino_score"] >= aq.SCORE_THR].copy().reset_index(drop=True)
    dataset["soft_label"] = list(np.array([v for v in dataset["soft_label"].values]))
    dataset["is_object"] = dataset["gdino_match"].astype(float)
    n_data = len(dataset)
    classes = sorted(dataset[aq.LABEL_COL].dropna().unique())
    S, _, _, _ = aq.set_graph(k, dataset, aq.DATA_SPACE, n_data=n_data)
    seed_indices = np.random.default_rng(aq.SEED).choice(n_data, size=aq.SEED_N, replace=False)
    if "seed_indices" in qh:
        assert np.array_equal(qh["seed_indices"], seed_indices), "seed set differs from the existing history"
    total_budget_n = max(1, int(n_data * aq.TARGET_BUDGET))

    for strategy in todo:
        t0 = time.time()
        order = [("seed", seed_indices.tolist())]

        def on_query(idx):
            order.append(("query", idx))
            if len(order) % 1000 == 1:
                print(f"  [{strategy}] {len(order) - 1} queries ({time.time() - t0:.0f}s)", flush=True)

        aq.run_active_learning_single(dataset, S, classes, k, alpha, total_budget_n, strategy,
                                      seed_indices, np.random.default_rng(aq.SEED), eval_every=10 ** 9,
                                      config=aq.AMGX_CONFIG, on_checkpoint=None, on_query=on_query)
        aq.save_query_history(OUT_PATH, strategy, order, k, alpha, seed_indices, n_data)
        print(f"{strategy} done: {len(order) - 1} queries [{time.time() - t0:.0f}s] -> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategies", nargs="+", default=aq.STRATEGIES, choices=aq.STRATEGIES)
    main(parser.parse_args().strategies)
