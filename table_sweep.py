# table_sweep.py
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
 
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

def summarize(values):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    mean = np.mean(values)
    sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0
    return mean, sem


all_results = {}   # param_name -> list of per-value summary dicts

total_runs = sum(len(v) for v in sweeps.values()) * n_replicates
# 4+4+3=11*10=110 runs

pbar = tqdm(total=total_runs, desc="sweeping kon/koff_target/k_slide_eff")

for param_name, param_values in sweeps.items():
    rows = []

    for value in param_values:
        target_occs, nontarget_occs, res_times, sites_visited, range_visited, t_conv = [], [], [], [], [], []

        for seed in range(n_replicates):
            full_params = {**base_config, param_name: value, "rng_seed": seed}
            out = run_OCC_SSA_tracked(**full_params)

            target_occs.append(analysis.target_occupancy(out))
            nontarget_occs.append(
                analysis.non_target_occupancy(out, base_config["M"], base_config["target_sites"])
            )
            res_times.append(analysis.mean_residence_time(out))
            sites_visited.append(analysis.mean_sites_visited(out))
            range_visited.append(analysis.mean_range_visited(out))

            t_conv.append(analysis.t_conv(out))

            pbar.set_postfix({param_name: value})
            pbar.update(1)

        row = {"value": value}
        row["target_occ_mean"], row["target_occ_sem"] = summarize(target_occs)
        row["nontarget_occ_mean"], row["nontarget_occ_sem"] = summarize(nontarget_occs)
        row["residence_time_mean"], row["residence_time_sem"] = summarize(res_times)
        row["sites_visited_mean"], row["sites_visited_sem"] = summarize(sites_visited)
        row["range_visited_mean"], row["range_visited_sem"] = summarize(range_visited)

        row["t_conv"], row["t_conv_sem"] = summarize(range_visited)

        rows.append(row)

    all_results[param_name] = rows

pbar.close()

# summary table for each swept parm using tabulate package
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
 