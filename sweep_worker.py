# table_sweep.py
import os
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from concurrent.futures import ProcessPoolExecutor, as_completed

from simulation_tracked import run_OCC_SSA_tracked
import analysis

base_config = dict(
    M=200,
    target_sites=[40, 50, 60, 140, 150, 160],
    Tmax=4.9e5, # 2e3 for some quick testing
    emit_every=500.0,
    koff_initial=0.12,
    koff_target=0.01,
    kon=6.28e-22,
    k_slide_eff=1e1,
)

n_replicates = 10

sweeps = {
    "kon": [6.28e-23, 6.28e-22, 6.28e-21, 6.28e-20],
    "koff_initial": [0.08, 0.12, 0.14],
    #"koff_target": [0.006, 0.01, 0.02, 0.06, 0.12],
    "k_slide_eff": [1e-1, 1, 10, 100],
}

# worker MUST be a top-level function (not nested) so it can be pickled and
# sent to separate worker processes -- runs one full replicate and returns
# just the metrics we need, computed inside the worker itself
def run_one_replicate(full_params):
    out = run_OCC_SSA_tracked(**full_params)
    return {
        "target_occ": analysis.target_occupancy(out),
        "nontarget_occ": analysis.non_target_occupancy(out, full_params["M"], full_params["target_sites"]),
        "residence_time": analysis.mean_residence_time(out),
        "sites_visited": analysis.mean_sites_visited(out),
        "range_visited": analysis.mean_range_visited(out),
        "t_conv": out["t_conv"],   # t_conv comes straight from the sim's own return dict, not analysis.py
    }


def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    mean = np.mean(values)
    sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sem


# multiprocessing needs this guard so worker processes don't re-run the
# whole script when they import it
if __name__ == "__main__":

    all_results = {}   # param_name -> list of per-value summary dicts

    total_runs = sum(len(v) for v in sweeps.values()) * n_replicates
    # 4+3+4=11*10=110 runs

    pbar = tqdm(total=total_runs, desc="sweeping kon/koff_initial/k_slide_eff")

    for param_name, param_values in sweeps.items():
        rows = []

        for value in param_values:
            target_occs, nontarget_occs, res_times = [], [], []
            sites_visited, range_visited, t_conv = [], [], []

            # build all n_replicates jobs for this parameter value up front
            jobs = [{**base_config, param_name: value, "rng_seed": seed} for seed in range(n_replicates)]

            # submit every job at once, collect results as they finish (any order)
            with ProcessPoolExecutor(max_workers=9) as pool:
                futures = {pool.submit(run_one_replicate, job): job["rng_seed"] for job in jobs}
                for fut in as_completed(futures):
                    metrics = fut.result()

                    target_occs.append(metrics["target_occ"])
                    nontarget_occs.append(metrics["nontarget_occ"])
                    res_times.append(metrics["residence_time"])
                    sites_visited.append(metrics["sites_visited"])
                    range_visited.append(metrics["range_visited"])
                    t_conv.append(metrics["t_conv"])

                    pbar.set_postfix({param_name: value})
                    pbar.update(1)

            row = {"value": value}
            row["target_occ_mean"], row["target_occ_sem"] = summarize(target_occs)
            row["nontarget_occ_mean"], row["nontarget_occ_sem"] = summarize(nontarget_occs)
            row["residence_time_mean"], row["residence_time_sem"] = summarize(res_times)
            row["sites_visited_mean"], row["sites_visited_sem"] = summarize(sites_visited)
            row["range_visited_mean"], row["range_visited_sem"] = summarize(range_visited)
            row["t_conv"], row["t_conv_sem"] = summarize(t_conv)   # fixed: was summarizing range_visited

            rows.append(row)

        all_results[param_name] = rows

    pbar.close()

    # summary table for each swept param using tabulate package
    for param_name, rows in all_results.items():
        print(f"\n=== sweeping {param_name} ===")
        table = [
            [
                r["value"],
                f"{r['target_occ_mean']:.4f} ± {r['target_occ_sem']:.4f}",
                f"{r['nontarget_occ_mean']:.4f} ± {r['nontarget_occ_sem']:.4f}",
                f"{r['residence_time_mean']:.3f} ± {r['residence_time_sem']:.3f}",
                f"{r['sites_visited_mean']:.3f} ± {r['sites_visited_sem']:.3f}",
                f"{r['range_visited_mean']:.3f} ± {r['range_visited_sem']:.3f}",
                f"{r['t_conv']:.3f} ± {r['t_conv_sem']:.3f}",
            ]
            for r in rows
        ]
        headers = ["value", "target_occ", "nontarget_occ", "residence_t", "sites_visited", "range_visited", "t_conv"]
        print(tabulate(table, headers=headers, tablefmt="simple"))