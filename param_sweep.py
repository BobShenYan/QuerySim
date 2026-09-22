# param_sweep.py
from tqdm import tqdm
from itertools import product
import numpy as np
import sim_io
from simulation import run_OCC_SSA

# everything that stays the same across the whole sweep
base_config = dict(
    M=200,
    target_sites=[40, 50, 60, 140, 150, 160],
    Tmax=2000.0,
    emit_every=500.0,
    rng_seed=42,  # same seed every combo
)

# only the parameters being tested, each with a list of values to try
sweep_params = {
    "koff_target": [0.005, 0.01, 0.02],
    "kon": [6.28e-22, 6.28e-21, 6.28e-20],
    # add more keys here later -- no other code needs to change
}

# build the grid: every combination of the swept parameters
names = list(sweep_params.keys())
value_lists = list(sweep_params.values())
all_combos = list(product(*value_lists))

pbar = tqdm(all_combos, desc="sweep progress")
for combo_values in pbar:
    combo = dict(zip(names, combo_values))

    # show which combo is currently running, plus tqdm's built-in elapsed/ETA
    pbar.set_postfix(combo)

    # merge base config with this combination's overrides
    full_params = {**base_config, **combo}

    # run the simulation with this combination
    results = run_OCC_SSA(**full_params)

    # build a filename from only swept params
    parts = [f"{key}{value}" for key, value in combo.items()]
    filename = "results/run_" + "_".join(parts) + ".npz"

    # save results + full params + seed, so this run is fully reproducible later
    sim_io.save_result(results, full_params, full_params["rng_seed"], filename)

    tqdm.write(f"saved {filename}")