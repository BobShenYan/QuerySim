# io.py, handling input/output
# corresponding results and parameters for long-term access
import os
import numpy as np

def save_result(results, params, seed, path):
    #Save a run_OCC_SSA results dict, plus the params/seed that produced it, to a single .npz file.
    
    # create results dierctory if it doesn't exist yet
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, results=results, params=params, seed=seed)

def load_result(path):
    #Load a previously saved result. Returns (results_dict, params_dict, seed).
    data = np.load(path, allow_pickle=True)
    results = data['results'].item()
    params = data['params'].item()
    seed = data['seed'].item()
    return results, params, seed
