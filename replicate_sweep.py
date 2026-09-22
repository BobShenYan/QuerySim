# replicate_sweep.py
# for cases where you want to have differing seeds across replicates

import numpy as np
from tqdm import tqdm
from simulation import run_OCC_SSA
import analysis
import plotting
import sim_io

base_config = dict(
    M=200,
    target_sites=[40, 50, 60, 140, 150, 160],
    Tmax=4.9e5,
    emit_every=500.0,
    koff_initial=0.12,
    kon=6.28e-22,
    k_slide_eff=1.0,
)

target_sites = np.asarray(base_config["target_sites"])
triplets = target_sites.reshape(-1, 3)

param_name = "koff_target"
param_values = [0.008, 0.01, 0.02, 0.04, 0.05, 0.06, 0.08, 0.12, 0.18]
n_replicates = 10

mean_ratios_raw = []
sem_ratios_raw = []
mean_ratios_strict = []
sem_ratios_strict = []
block_freqs = []

total_runs = len(param_values) * n_replicates
pbar = tqdm(total=total_runs, desc="sweeping")

for value in param_values:
    per_replicate_raw = []
    per_replicate_strict = []
    per_replicate_blocked = []

    for seed in range(n_replicates):
        full_params = {**base_config, param_name: value, "rng_seed": seed}
        out = run_OCC_SSA(**full_params)

        filename = f"results/replicate_{param_name}{value}_seed{seed}.npz"
        sim_io.save_result(out, full_params, seed, filename)

        triplet_raw = []
        triplet_strict = []
        triplet_blocked = []

        for left, center, right in triplets:
            metrics = analysis.compute_blocking_metric(
                out, target_sites, flanked_site=center, edge_sites=(left, right),
                profile_mode="all",
            )
            triplet_raw.append(metrics["center_ratio_raw"])
            triplet_strict.append(metrics["center_ratio_strict"])
            triplet_blocked.append(metrics["blocked_strict"])

        per_replicate_raw.append(np.mean(triplet_raw))
        per_replicate_strict.append(np.nanmean(triplet_strict))
        per_replicate_blocked.append(np.mean(triplet_blocked))

        pbar.set_postfix({param_name: value})
        pbar.update(1)

    per_replicate_raw = np.array(per_replicate_raw)
    per_replicate_strict = np.array(per_replicate_strict)
    per_replicate_blocked = np.array(per_replicate_blocked)

    mean_ratios_raw.append(np.mean(per_replicate_raw))
    sem_ratios_raw.append(np.std(per_replicate_raw, ddof=1) / np.sqrt(len(per_replicate_raw)))

    valid_strict = per_replicate_strict[~np.isnan(per_replicate_strict)]
    if len(valid_strict) > 0:
        mean_ratios_strict.append(np.mean(valid_strict))
        sem_ratios_strict.append(np.std(valid_strict, ddof=1) / np.sqrt(len(valid_strict)) if len(valid_strict) > 1 else 0.0)
    else:
        mean_ratios_strict.append(np.nan)
        sem_ratios_strict.append(np.nan)

    block_freqs.append(np.mean(per_replicate_blocked))

pbar.close()

# save file still work in progress
plotting.plot_ratio_vs_param(
    param_values, mean_ratios_raw, sem_ratios_raw,
    xlabel=param_name,
    title=f"Center/flanking ratio (raw) vs {param_name}",
    save_to=f"figures/ratio_raw_vs_{param_name}.png",
)

plotting.plot_ratio_vs_param(
    param_values, mean_ratios_strict, sem_ratios_strict,
    xlabel=param_name,
    title=f"Center/flanking ratio (strict) vs {param_name}",
    save_to=f"figures/ratio_strict_vs_{param_name}.png",
)

plotting.plot_block_frequency(
    param_values, block_freqs,
    xlabel=param_name,
    title=f"Strict blocking frequency vs {param_name}",
    save_to=f"figures/block_freq_vs_{param_name}.png",
)

print(f"\n{'value':<10} {'mean_raw':<12} {'mean_strict':<14} {'block_freq':<10}")
for v, mr, ms, bf in zip(param_values, mean_ratios_raw, mean_ratios_strict, block_freqs):
    ms_str = "nan" if np.isnan(ms) else f"{ms:.4f}"
    print(f"{v:<10} {mr:<12.4f} {ms_str:<14} {bf:<10.3f}")