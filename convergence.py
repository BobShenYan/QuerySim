# convergence.py
# tracking and measuring convergence via MSD
# biologically stable transcription @ equillibrium?

import numpy as np

# convergence testing after all profile_series data is completely collected

# all site convergence
def compute_chi2_series(profile_series):
    diffs = (profile_series[1:]-profile_series[:-1])
    chi2 = np.mean(diffs**2, axis = 1)
    return chi2

# target site convergence
def compute_tchi2_series(profile_series,target_sites):
    targ_profiles = profile_series[:, target_sites]
    diffs = (targ_profiles[1:]-targ_profiles[:-1])
    tchi2 = np.mean(diffs**2, axis = 1)
    return tchi2

# all sites
def estimate_conv_time(chi2, profile_times, window=20, tail_frac=0.8, k_consecutive=10):
    chi2_times = profile_times[1:]

    kernel = np.ones(window) / window

    # what does convolve do again
    chi_smooth = np.convolve(chi2, kernel, mode="same")

    tail_start = int(tail_frac * len(chi_smooth))
    chi_tail = chi_smooth[tail_start:]

    baseline = np.mean(chi_tail)
    spread = np.std(chi_tail)
    threshold = baseline + spread

    count = 0
    t_conv = None

    # no separation between times and values
    for i, val in enumerate(chi_smooth):
        if val <= threshold:
            count += 1
            if count >= k_consecutive:
                t_conv = chi2_times[i - k_consecutive + 1]
                break
        else:
            count = 0
    return baseline, spread, threshold, t_conv