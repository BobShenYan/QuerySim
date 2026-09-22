# plotting.py
import numpy as np
import matplotlib.pyplot as plt
 
 
def plot_profile(result, save_to=None, title=None):
    # Spatial occupancy profiles over time, converging toward steady state
    profiles = result["profile_series"]
    target_sites = result.get("target_sites")
 
    plt.figure(figsize=(7, 4))
    for i, prof in enumerate(profiles):
        alpha = 0.15 if i < len(profiles) - 5 else 0.6
        plt.plot(prof, color="black", alpha=alpha)
 
    if target_sites is not None:
        for ts in target_sites:
            plt.axvline(ts, color="red", linestyle="--", alpha=0.4)
 
    plt.xlabel("DNA position")
    plt.ylabel("occupancy (weighted)")
    plt.title(title or "Occupancy profiles converging to steady-state")
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()
 
 
def plot_occfrac(result, save_to=None, title=None):
    # Overall DNA occupancy fraction (n/M) over time
    plt.figure(figsize=(10, 5))
    plt.plot(result["occfrac_times"], result["occfrac_series"], lw=2)
    plt.xlabel("time (s)")
    plt.ylabel("occupancy fraction n/M")
    plt.title(title or "Total DNA occupancy over time")
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()
 
 
def plot_occtarg(result, save_to=None, title=None):
    # Target-site occupancy fraction over time
    plt.figure(figsize=(10, 5))
    plt.plot(result["occtarg_times"], result["occtarg_series"], lw=2)
    plt.xlabel("time (s)")
    plt.ylabel("occupancy fraction num_avg/N_target")
    plt.title(title or "Target occupancy over time")
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()
 
 
def plot_chi2(chi2, chi2_times, save_to=None, title=None):
    # Chi-squared convergence diagnostic over time (from convergence.py's output)
    plt.figure(figsize=(10, 6))
    plt.plot(chi2_times, chi2, lw=2)
    plt.xlabel("time (s)")
    plt.ylabel("mean squared change")
    plt.title(title or "Occupancy convergence")
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()

def plot_chi2_comparison(chi2_all, chi2_targ, chi2_times, save_to=None, title=None):
    # Overlay non-specific (all-site) and target-only chi-squared convergence curves
    plt.figure(figsize=(10, 6))
    plt.plot(chi2_times, chi2_all, label="non-specific", lw=1.5)
    plt.plot(chi2_times, chi2_targ, label="target only", lw=1.5)
    plt.xlabel("time (s)")
    plt.ylabel("mean squared error")
    plt.title(title or "Target and non-specific occupancy convergence")
    plt.legend()
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()

def plot_ratio_vs_param(param_values, mean_ratios, sem_ratios, xlabel, title=None, save_to=None):
    # Plot mean center/flanking ratio (with SEM error bars) against a swept
    # parameter, using evenly-spaced categorical x-axis positions labeled with
    #the actual parameter values.

    x_positions = np.arange(len(param_values))

    plt.figure(figsize=(8, 5))
    plt.errorbar(x_positions, mean_ratios, yerr=sem_ratios, marker="o", color="black", capsize=3)
    plt.xticks(x_positions, [str(v) for v in param_values], rotation=45)
    plt.xlabel(xlabel)
    plt.ylabel("center / flanking occupancy ratio")
    plt.title(title or f"Center/flanking ratio vs {xlabel}")
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()

# Fraction of replicates showing strict blocking, vs a swept parameter
def plot_block_frequency(param_values, block_freqs, xlabel, title=None, save_to=None):
    x_positions = np.arange(len(param_values))
    plt.figure(figsize=(7, 4))
    plt.plot(x_positions, block_freqs, marker="o", color="black")
    plt.xticks(x_positions, [str(v) for v in param_values], rotation=45)
    plt.ylim(-0.05, 1.05)
    plt.xlabel(xlabel)
    plt.ylabel("strict blocking frequency")
    plt.title(title or f"Strict blocking frequency vs {xlabel}")
    plt.tight_layout()
    if save_to:
        plt.savefig(save_to)
        plt.close()
    else:
        plt.show()