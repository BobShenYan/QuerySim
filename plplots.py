# PLPlots.py: plotting, loading, and promoting figures

import glob_plt
import promote_figure
import plotting
import os
import sim_io
import numpy as np

# manually set
run_name = "blockrepwork_Tmax4.9e5"

target_sites = np.array([40, 50, 60, 140, 150, 160])
triplets = target_sites.reshape(-1, 3) 
param_name = "kon"
sweep_value = 6.28e-20
seed = 0

result, means, sems = glob_plt.reload_sweep_curve(
    param_name, triplets, target_sites, metric="ratio_raw"
)
print(result, means, sems)  

fig_dir = f"figures/{run_name}" # where you are saving into
os.makedirs("figures", exist_ok=True)


one_file = f"results/{run_name}/{param_name}/table_*_{sweep_value}_*_seed{seed}.npz"
result, params, seed = sim_io.load_result(one_file)

plotting.plot_profile(result, save_to=f"{fig_dir}/{run_name}/{param_name,sweep_value}_seed{seed}.png", title = "occupancy profile")