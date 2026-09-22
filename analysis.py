# analysis.py
import numpy as np
import convergence

def final_target_occupancy(result):
    # Mean target-site occupancy fraction over the last few emitted windows
    return np.mean(result["occtarg_series"][-5:])
 
def final_global_occupancy(result):
    # Mean overall (n/M) occupancy fraction over the last few emitted windows
    return np.mean(result["occfrac_series"][-5:])
 
def final_profile(result):
    # The last recorded spatial occupancy profile (length-M array)
    return result["profile_series"][-1]
 
def summarize_run(result, params):
    # One-line summary dict for a single run -- convenient for building a table across a sweep.
    return {
        "params": params,
        "final_target_occ": final_target_occupancy(result),
        "final_global_occ": final_global_occupancy(result),
        "events": result["events"], # for cases where you want to extract from 
        "end_t": result["end_t"],
    }

def summarize_sweep(loaded_runs):
    # Given a list of (result, params, seed) tuples (e.g. from loading every
    # file in a sweep's results/ folder), return a list of summary dicts,
    # one per run -- suitable for building a comparison table or dataframe
    return [summarize_run(result, params) for result, params, seed in loaded_runs]

 
def compare_final_profiles(result_a, result_b):
    # Elementwise difference between two runs' final spatial occupancy profiles
    return final_profile(result_a) - final_profile(result_b)


def compute_center_flanking_ratio(profile, target_sites):
    # For a single spatial occupancy profile, compute the center-to-flanking
    # occupancy ratio for each (left_flank, center, right_flank) triplet in
    # target_sites, then average across all triplets
 
    target_sites = np.asarray(target_sites)
    if target_sites.size % 3 != 0:
        raise ValueError(
            f"target_sites must be structured in groups of 3 (flank, center, flank); "
            f"got {target_sites.size} sites, not divisible by 3."
        )

    # -1 for automatic determination
    triplets = target_sites.reshape(-1, 3)
    left_flank = triplets[:, 0]
    center = triplets[:, 1]
    right_flank = triplets[:, 2]

    center_occ = profile[center]
    flank_occ = (profile[left_flank] + profile[right_flank]) / 2.0

    per_triplet_ratios = center_occ / flank_occ
    mean_ratio = np.mean(per_triplet_ratios)

    return per_triplet_ratios, mean_ratio

# a detailed approach to track blocking
def compute_blocking_metric(result, target_sites, flanked_site, edge_sites,
                              t_start=None, profile_mode="all", last_n=5):
    selected_profiles, selected_times, t_start = get_post_convergence_profiles(
        result, target_sites, t_start=t_start, profile_mode=profile_mode, last_n=last_n,
    )

    mean_profile = np.mean(selected_profiles, axis=0)

    left = float(mean_profile[edge_sites[0]])
    mid = float(mean_profile[flanked_site])
    right = float(mean_profile[edge_sites[1]])

    blocked_strict = (mid < left) and (mid < right)
    blocked_weak = (mid < left) or (mid < right)

    flank_mean = (left + right) / 2.0
    center_ratio_raw = mid / flank_mean if flank_mean > 0 else np.nan
    center_ratio_strict = center_ratio_raw if blocked_strict else np.nan
    asymmetry = abs(left - right)

    return {
        "t_start": float(t_start),
        "flanked_site": int(flanked_site),
        "edge_sites": tuple(int(x) for x in edge_sites),
        "left_occ": left,
        "mid_occ": mid,
        "right_occ": right,
        "blocked_strict": blocked_strict,
        "blocked_weak": blocked_weak,
        "center_ratio_raw": center_ratio_raw,
        "center_ratio_strict": center_ratio_strict,
        "asymmetry": asymmetry,
        "n_profiles_used": int(len(selected_profiles)),
        "mean_profile": mean_profile,
    }


def center_flanking_ratio_over_replicates(profiles, target_sites):
    # Given a list of final profiles (one per replicate/seed, same parameters),
    # return (mean_ratio, sem_ratio, per_replicate_ratios) across replicates
  
    per_replicate_ratios = []
    for profile in profiles:
        _, mean_ratio = compute_center_flanking_ratio(profile, target_sites)
        per_replicate_ratios.append(mean_ratio)

    per_replicate_ratios = np.array(per_replicate_ratios)
    mean_ratio = np.mean(per_replicate_ratios)
    sem_ratio = np.std(per_replicate_ratios, ddof=1) / np.sqrt(len(per_replicate_ratios))

    return mean_ratio, sem_ratio, per_replicate_ratios

# retrieves profiles after convergence time t_conv
def get_post_convergence_profiles(result, target_sites, t_start=None, profile_mode="all", last_n=5):
    profiles = result["profile_series"]
    times = result["profile_times"]

    if len(profiles) == 0:
        raise ValueError("No emitted profiles found in result.")

    if t_start is None:
        chi2_targ = convergence.compute_tchi2_series(profiles, target_sites)
        _, _, _, t_start = convergence.estimate_conv_time(chi2_targ, times)

    # if the run wasn't able to converge, show error message
    if t_start is None:
        raise ValueError("Could not determine convergence time. Try a longer simulation.")

    mask = times >= t_start
    profiles_post = profiles[mask]
    times_post = times[mask]

    if len(profiles_post) == 0:
        raise ValueError("No profiles available after convergence time.")

    if profile_mode == "all":
        return profiles_post, times_post, float(t_start)
    if profile_mode == "last_n":
        n_use = min(last_n, len(profiles_post))
        return profiles_post[-n_use:], times_post[-n_use:], float(t_start)
    raise ValueError("profile_mode must be 'all' or 'last_n'")


def non_target_occupancy(result, M, target_sites):
    N_target = len(target_sites)
    occfrac = result["occfrac_series"][-1]
    occtarg = result["occtarg_series"][-1]

    n_bound_total = occfrac * M
    n_bound_target = occtarg * N_target
    n_bound_nontarget = n_bound_total - n_bound_target

    return n_bound_nontarget / (M - N_target)


def target_occupancy(result):
    return result["occtarg_series"][-1]


def mean_residence_time(result):
    rt = result["residence_times"]
    return np.mean(rt) if len(rt) > 0 else np.nan


def mean_sites_visited(result):
    sv = result["sites_visited_counts"]
    return np.mean(sv) if len(sv) > 0 else np.nan


def mean_range_visited(result):
    rv = result["range_visited"]
    return np.mean(rv) if len(rv) > 0 else np.nan