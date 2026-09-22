# make_figures.py

import glob
import os # for file creation
import sim_io # accessing results
import plotting

# exist_ok, no error if folder already exists
os.makedirs("figures", exist_ok=True)

# manually input to match sweep_params dict used to create custom file name
swept_keys = ("koff_target", "kon")

# all files in results sorted
for path in sorted(glob.glob("results/*.npz")):
    result, params, seed = sim_io.load_result(path)
    tag_parts = [f"{key}{params[key]}" for key in swept_keys] # key from swept_keys and associated value
    tag = "_".join(tag_parts) # join together with underscore
    plotting.plot_profile(result, save_to=f"figures/profile_{tag}.png") #
    plotting.plot_chi2_comparison(result, save_to=f"figures/occtarg_{tag}.png") #