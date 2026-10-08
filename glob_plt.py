# import glob
# import sim_io
# import analysis

#import plotting
# # the point is that now we can load previous runs from /results and perform analysis on prior data

# # load every saved run back in
# loaded_runs = []
# for path in sorted(glob.glob("results/*.npz")):
#     result, params, seed = sim_io.load_result(path)
#     loaded_runs.append((result, params, seed))

# # get a quick summary table across the whole sweep 
# summaries = analysis.summarize_sweep(loaded_runs)
# for s in summaries:
#     print(s)

# result, params, seed = sim_io.load_result("results/run_koff_target0.02_kon6.28e-20.npz")
# plotting.plot_profile(result, save_to="figures/profile_kt0.02_kon6.28e-20.png")


import glob
import numpy as np
import sim_io
import analysis
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# manually inputting run_name for now, needs to be synced with repwork_sweep.py:
# run_name = "blockrepwork_Tmax4.9e5"

def reload_sweep_curve(run_name, param_name, triplets, target_sites, metric="ratio_raw"):
    paths = sorted(glob.glob(f"results/{run_name}/{param_name}/table_*_seed*.npz"))
    paths = [p for p in paths if ".tmp" not in p]

    by_value = {}
    for path in paths:
        result, params, seed = sim_io.load_result(path)
        value = params[param_name]
        by_value.setdefault(value, []).append(result)

    param_values, means, sems = [], [], []
    for value in sorted(by_value.keys()):
        per_replicate = []
        for result in by_value[value]:
            triplet_vals = []
            for left, center, right in triplets:
                try:
                    m = analysis.compute_blocking_metric(
                        result, target_sites, flanked_site=center, edge_sites=(left, right),
                        profile_mode="all",
                    )
                    if metric == "ratio_raw":
                        triplet_vals.append(m["center_ratio_raw"])
                    elif metric == "ratio_strict":
                        triplet_vals.append(m["center_ratio_strict"])
                    elif metric == "block_freq":
                        triplet_vals.append(m["blocked_strict"])
                except ValueError:
                    triplet_vals.append(np.nan)
            per_replicate.append(np.nanmean(triplet_vals))

        per_replicate = np.array(per_replicate, dtype=float)
        per_replicate = per_replicate[~np.isnan(per_replicate)]
        param_values.append(value)
        means.append(np.mean(per_replicate) if len(per_replicate) else np.nan)
        sems.append(np.std(per_replicate, ddof=1)/np.sqrt(len(per_replicate)) if len(per_replicate) > 1 else 0.0)

    return param_values, means, sems


def plot_two_param_comparison(curve_a, curve_b, label_a, label_b, ylabel, save_to=None, title=None):
    fig, ax = plt.subplots(figsize=(9, 5))

    for (values, means, sems), label in [(curve_a, label_a), (curve_b, label_b)]:
        x = np.arange(len(values))
        ax.errorbar(x, means, yerr=sems, marker="o", capsize=3, label=f"{label}: {values}")

    #ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    #ax.set_title(title or "Comparison across two swept parameters")
    ax.legend(fontsize=8)
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()

def reload_p_given_curve(param_name, triplet_index):
    paths = sorted(glob.glob(f"results/{run_name}/{param_name}/table_*_seed*.npz"))
    paths = [p for p in paths if ".tmp" not in p]

    by_value = {}
    for path in paths:
        result, params, seed = sim_io.load_result(path)
        value = params[param_name]
        by_value.setdefault(value, []).append(result["p_center_given_flanks"][triplet_index])

    param_values = sorted(by_value.keys())
    means, sems = [], []
    for v in param_values:
        vals = np.array(by_value[v], dtype=float)
        vals = vals[~np.isnan(vals)]
        means.append(np.mean(vals) if len(vals) else np.nan)
        sems.append(np.std(vals, ddof=1)/np.sqrt(len(vals)) if len(vals) > 1 else 0.0)

    return param_values, means, sems

# param_name = "koff_target"
# t0_curve = reload_p_given_curve(param_name, triplet_index=0)
# t1_curve = reload_p_given_curve(param_name, triplet_index=1)

# plot_two_param_comparison(
#     t0_curve, t1_curve,
#     label_a="triplet 0", label_b="triplet 1",
#     ylabel="p_center_given_flanks",
#     save_to=f"figures/p_given_t0_vs_t1_{param_name}.png",
#     title=f"p_center_given_flanks: triplet 0 vs triplet 1 (sweeping {param_name})",
# )

def reload_metric_curve(param_name, metric_name):
    metric_funcs = {
        "target_occ": analysis.target_occupancy,
        "residence_time": analysis.mean_residence_time,
        "sites_visited": analysis.mean_sites_visited,
        "range_visited": analysis.mean_range_visited,
        "t_conv": lambda r: r["t_conv"],
    }
    func = metric_funcs[metric_name]

    paths = sorted(glob.glob(f"results/{run_name}/{param_name}/table_*_seed*.npz"))
    paths = [p for p in paths if ".tmp" not in p]

    by_value = {}
    for path in paths:
        result, params, seed = sim_io.load_result(path)
        value = params[param_name]
        by_value.setdefault(value, []).append(func(result))

    param_values = sorted(by_value.keys())
    means, sems = [], []
    for v in param_values:
        vals = np.array(by_value[v], dtype=float)
        vals = vals[~np.isnan(vals)]
        means.append(np.mean(vals) if len(vals) else np.nan)
        sems.append(np.std(vals, ddof=1)/np.sqrt(len(vals)) if len(vals) > 1 else 0.0)

    return param_values, means, sems