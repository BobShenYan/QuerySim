import os
import numpy as np
from tqdm import tqdm
from tabulate import tabulate
from concurrent.futures import ProcessPoolExecutor, as_completed

from simulation_tracked import run_OCC_SSA_tracked
import analysis
import sim_io
import glob

base_config = dict(
    M=200,
    target_sites=[40, 50, 60, 140, 150, 160], # 2 triplets: (40,50,60), (140,150,160)
    Tmax=4.9e5, #49000
    emit_every=500.0,
    koff_initial=0.12,
    koff_target=0.01,
    kon=6.28e-22,
    kon_target_factor=1.0,
    slide_to_targ=1.0,
    k_slide_eff=10.0,
    redirect="off",
)

# manual naming option, for config changes, while still keeping prior config data
run_name = "kon_target_repwork_Tmax4.9e5"

target_sites = np.asarray(base_config["target_sites"])
triplets = target_sites.reshape(-1, 3)
n_triplets = triplets.shape[0] # (2 rows, 3 cols) [0]

sweeps = {
    #"kon": [6.28e-23, 6.28e-22, 6.28e-21, 6.28e-20],
    "koff_target": [0.006, 0.01, 0.02, 0.06, 0.12],
    "k_slide_eff": [1e-1, 1.0, 0, 10, 100],
}
n_replicates = 30

def run_one_replicate(full_params, param_name): # unpack the specific config for each rep

    out = None

    tag = "_".join(f"{k}{v}" for k, v in full_params.items() if k in ("kon", "koff_target", "k_slide_eff"))
    filename = f"results/{run_name}/{param_name}/table_{tag}_seed{full_params['rng_seed']}.npz" # jobs final filename

    # file config already exists
    if os.path.exists(filename): 
        loaded, params, _ = sim_io.load_result(filename)
        if params == full_params:
            out = loaded

    # new config, needs sim
    if out is None:
        out = run_OCC_SSA_tracked(**full_params)
        sim_io.save_result(out, full_params, full_params["rng_seed"], filename)
    
    # raw: 
    triplet_raw, triplet_strict, triplet_blocked = [], [], []
    for left, center, right in triplets:
        try:
            m = analysis.compute_blocking_metric(
                out, target_sites, flanked_site=center, edge_sites=(left, right),
                profile_mode="all",
            )
            triplet_raw.append(m["center_ratio_raw"])
            triplet_strict.append(m["center_ratio_strict"])
            triplet_blocked.append(m["blocked_strict"])
        except ValueError:
            triplet_raw.append(np.nan)
            triplet_strict.append(np.nan)
            triplet_blocked.append(False)

    result = {
        "target_occ": analysis.target_occupancy(out),
        "nontarget_occ": analysis.non_target_occupancy(out, base_config["M"], base_config["target_sites"]),
        "residence_time": analysis.mean_residence_time(out),
        "sites_visited": analysis.mean_sites_visited(out),
        "range_visited": analysis.mean_range_visited(out),
        "t_conv": out["t_conv"],
        "time_center_given_flanks": out["time_center_given_flanks"],
        "ratio_raw": np.mean(triplet_raw),
        "ratio_strict": np.nanmean(triplet_strict),
        "block_freq": np.mean(triplet_blocked),
        #"p_center_given_flanks": analysis.mean_p_center_given_flanks(out),
        #"p_center_unconditional": analysis.mean_p_center_unconditional(out),
        "out": out,          # keeping the raw result metrics so we can save it and use it in the future
        "full_params": full_params,
    }

    # per-triplet instead of averaged
    for i in range(n_triplets):
        result[f"p_given_t{i}"] = out["p_center_given_flanks"][i]
        result[f"p_uncond_t{i}"] = out["p_center_unconditional"][i]

    return result

def summarize(values): # wrapper for processing table outputs
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    mean = np.mean(values)
    sem = np.std(values, ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0.0 # let's use SEM specifically for mean calcs
    return mean, sem

if __name__ == "__main__":
    all_results = {}
    total_runs = sum(len(v) for v in sweeps.values()) * n_replicates
    pbar = tqdm(total=total_runs, desc="sweeping kon/koff_target/k_slide_eff")

    # leaving a couple cores free
    max_workers = max(1, os.cpu_count() - 2) # saftety net of 1 core

    metric_names = ["target_occ", "nontarget_occ", "residence_time", "sites_visited",
                     "range_visited", "t_conv", "time_center_given_flanks", "ratio_raw", "ratio_strict", "block_freq"]
    for i in range(n_triplets):
        metric_names += [f"p_given_t{i}", f"p_uncond_t{i}"]

    for param_name, param_values in sweeps.items():
        rows = []

        for value in param_values:
            collected = {k: [] for k in metric_names}

            jobs = [{**base_config, param_name: value, "rng_seed": seed} for seed in range(n_replicates)]

            with ProcessPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(run_one_replicate, job, param_name): job["rng_seed"] for job in jobs}
                for fut in as_completed(futures):
                    try:
                        metrics = fut.result()
                    except Exception as e:
                        print(f"FAILED {param_name}={value} seed={futures[fut]}: {e!r}") # print repr for debugging
                        pbar.update(1)
                        continue

                    for k in metric_names:
                        collected[k].append(metrics[k])

                    pbar.set_postfix({param_name: value})
                    pbar.update(1)

            row = {"value": value}
            for k in metric_names:
                row[f"{k}_mean"], row[f"{k}_sem"] = summarize(collected[k])
            rows.append(row)

        all_results[param_name] = rows

    pbar.close()

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
                f"{r['t_conv_mean']:.3f} ± {r['t_conv_sem']:.3f}",
                f"{r['time_center_given_flanks_mean']:.3f} ± {r['time_center_given_flanks_sem']:.3f}",
                f"{r['ratio_raw_mean']:.4f} ± {r['ratio_raw_sem']:.4f}",
                f"{r['ratio_strict_mean']:.4f} ± {r['ratio_strict_sem']:.4f}",
                f"{r['block_freq_mean']:.3f} ± {r['block_freq_sem']:.3f}",
            ] + [
                f"{r[f'p_given_t{i}_mean']:.4f} ± {r[f'p_given_t{i}_sem']:.4f}"
                for i in range(n_triplets)
            ] + [
                f"{r[f'p_uncond_t{i}_mean']:.4f} ± {r[f'p_uncond_t{i}_sem']:.4f}"
                for i in range(n_triplets)
            ]
            for r in rows
        ]
        headers = ["value", "target_occ", "nontarget_occ", "residence_t", "sites_visited",
                   "range_visited", "t_conv","time_center_given_flanks", "ratio_raw", "ratio_strict", "block_freq"]
        headers += [h for i in range(n_triplets) for h in (f"p_given_t{i}", f"p_uncond_t{i}")]
        table = tabulate(table, headers=headers, tablefmt="simple")
        print(table)

        os.makedirs(f"results/{run_name}", exist_ok=True)
        with open(f"results/{run_name}/table_{param_name}.txt", "w", encoding="utf-8") as file:
            file.write(table + "\n")


