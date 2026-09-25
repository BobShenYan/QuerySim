# simulation_tracked.py
from kinetics import compute_spawn_rate, compute_off_rate, compute_slide_rate
from observables import Observer

import convergence
#from convergence import compute_chi2_series, estimate_conv_time

from concurrent.futures import ProcessPoolExecutor, as_completed # parrallel processing

import numpy as np

# specific to range_visited
def _circular_span(visited, M):
    # Works by finding the single largest GAP between consecutive visited positions (going around the circle) the span is everything except that gap 
    positions = sorted(visited)
    if len(positions) <= 1:
        return 0

    gaps = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    wraparound_gap = (M - positions[-1]) + positions[0]
    gaps.append(wraparound_gap)

    largest_gap = max(gaps)
    return M - largest_gap

def run_OCC_SSA_tracked(
    M=200,
    target_sites=None,
    k_slide_eff=1e1,
    koff_initial=0.12,
    koff_target=0.01,
    kon=6.28e-21,
    TF_conc=1e-9,
    N_A=6.022e26,
    Tmax=10000.0,
    max_events=int(1e10),
    rng_seed=None,
    c_on=1.0,
    c_off=1.0,
    p_rebind=0.0,
    rebind_radius=5,
    rebind_tries=10,
    emit_every=500.0
    ):

    t = 0.0

    rng = np.random.default_rng(rng_seed)

    if target_sites is None:
        target_sites = np.linspace(M // 7, 6 * M // 7, 3, dtype=int)
    target_sites = np.unique(np.asarray(target_sites, dtype=int))
    is_target = np.zeros(M, dtype=bool)
    is_target[target_sites] = True

    triplets = target_sites.reshape(-1, 3) #
    n_triplets = triplets.shape[0] #
    time_flanks_occupied = np.zeros(n_triplets)
    time_center_given_flanks = np.zeros(n_triplets)
    time_center_total = np.zeros(n_triplets)

    tf_positions = np.empty(M, dtype=np.int64)   # pre-allocating space of M slots
    tf_birth = np.empty(M, dtype=np.float64)     # when each currently-bound TF attached
    tf_visited = []                               # set of distinct sites visited so far
    n = 0

    occ = np.zeros(M, dtype=bool)
    target_count = 0
    n_tf_on_target = 0

    hop_rate = float(k_slide_eff)

    track1 = Observer(M, target_sites, emit_every)

    residence_times = []   # completed residence durations, appended on every unbind
    sites_visited_counts = []   # number of distinct sites visited, appended on every unbind
    range_visited_list = []   # circular span (max extent) of sites visited, appended on every unbind

    for ev in range(max_events):
        if t >= Tmax:
            break

        #n = tf_positions.size
        num_bound = target_count

        empty_sites = M - n

        n_on_target = n_tf_on_target
        n_off_target = n - n_on_target

        Rspawn = compute_spawn_rate(kon, TF_conc, N_A, empty_sites, c_on, num_bound)
        Roff, koff_t, koff_i, slide_scale_target = compute_off_rate(n_on_target, n_off_target, koff_target, koff_initial, num_bound, c_off)
        Rslide = compute_slide_rate(n_on_target, n_off_target, hop_rate, koff_t, koff_i)

        R_tot = Rspawn + Rslide + Roff

        dt = rng.exponential(1/R_tot)

        t_old = t
        t += dt

        track1.record(tf_positions[:n], n, num_bound, dt, t)

        for i, (left, center, right) in enumerate(triplets):
            if occ[left] and occ[right]:
                time_flanks_occupied[i] += dt
                if occ[center]:
                    time_center_given_flanks[i] += dt # implement a table outputin sweep, this is also equivalent to transcription
            if occ[center]:
                time_center_total[i] += dt

        u = rng.random() * R_tot

        # spawn
        if u < Rspawn:
            while True:
                pos = rng.integers(0, M)
                if not occ[pos]:
                    break

            tf_positions[n] = pos
            tf_birth[n] = t
            n += 1

            tf_visited.append({int(pos)})

            occ[pos] = True
            if is_target[pos]:
                target_count += 1
                n_tf_on_target += 1

        # off/unbinding
        elif u < Rspawn + Roff:
            if n == 0:
                continue

            on_t = is_target[tf_positions[:n]]
            weights = np.where(on_t, koff_t, koff_i)
            wsum = weights.sum()
            if wsum <= 0:
                continue

            # choosing to run 3 parrallel arrays so that each TF wouldn't need to be treated as an object, I'd lose vectorization functionality
            idx = rng.choice(n, p=weights / wsum)
            old_pos = tf_positions[idx]
            birth_t = tf_birth[idx] # store in memory the t of this specific idx TF as birth_t
            visited = tf_visited[idx]

            # remove by swap-with-last across all three parallel arrays
            last = n - 1
            tf_positions[idx] = tf_positions[last]
            tf_birth[idx] = tf_birth[last] 
            n -= 1

            tf_visited[idx] = tf_visited[last]
            tf_visited.pop()

            occ[old_pos] = False

            if is_target[old_pos]:
                target_count -= 1
                n_tf_on_target -= 1

            rebound = False
            if rng.random() < p_rebind and tf_positions.size < M:
                for _ in range(rebind_tries):
                    cand = (old_pos + rng.integers(-rebind_radius, rebind_radius + 1)) % M
                    if not occ[cand]:
                        tf_positions[n] = cand
                        tf_birth[n] = t    # residence restarts at rebind site
                        tf_visited.append({int(cand)})
                        occ[cand] = True
                        if is_target[cand]:
                            target_count += 1
                            n_tf_on_target += 1
                        rebound = True
                        break

            # record the completed residence event that just ended
            residence_times.append(t - birth_t)
            sites_visited_counts.append(len(visited))
            range_visited_list.append(_circular_span(visited, M))

        # slide
        else:
            if n == 0:
                continue

            on_t = is_target[tf_positions[:n]]
            slide_weights = np.where(on_t, hop_rate * slide_scale_target, hop_rate)
            wsum_slide = slide_weights.sum()
            if wsum_slide <= 0:
                continue

            idx = rng.choice(n, p=slide_weights / wsum_slide)
            old_pos = tf_positions[idx]
            occ[old_pos] = False

            if is_target[old_pos]:
                target_count -= 1
                n_tf_on_target -= 1

            step = -1 if rng.random() < 0.5 else 1
            cand = (old_pos + step) % M
            if occ[cand]:
                alt = (old_pos - step) % M
                cand = alt if not occ[alt] else old_pos

            tf_positions[idx] = cand
            tf_visited[idx].add(int(cand))   # sets automatically ignore duplicates
            occ[cand] = True

            if is_target[cand]:
                target_count += 1
                n_tf_on_target += 1

    results = track1.get_results()

    # extracting convergence values (t_conv)
    profiles = results["profile_series"]
    times = results["profile_times"]

    tchi2_targ = convergence.compute_tchi2_series(profiles,target_sites)
    _, _, _, t_conv = convergence.estimate_conv_time(tchi2_targ, times)

    # chi2_targ = convergence.compute_chi2_series(profiles)
    # _, _, _, t_conv = convergence.estimate_conv_time(chi2_targ, times)

    # filter out None cases
    converged = t_conv is not None

    # don't use raise, othewise script will crash
    if not converged:
        print(f"Could not determine convergence time for koff_targt {koff_target} with seed {rng_seed}. Try different config or a longer simulation.")

    p_center_given_flanks = np.where(
        time_flanks_occupied > 0,
        time_center_given_flanks / np.where(time_flanks_occupied > 0, time_flanks_occupied, 1),
        np.nan,
    )
    p_center_unconditional = time_center_total / t

    return {
        "end_t": t,
        "tf_positions": tf_positions[:n], # recall that only n positions are filled with n TFs total
        "occ": occ,
        "target_count": target_count,
        "n_tf_on_target": n_tf_on_target,
        "events": ev + 1,
        "residence_times": np.array(residence_times),
        "sites_visited_counts": np.array(sites_visited_counts),
        "range_visited": np.array(range_visited_list),
        "t_conv": t_conv,
        "p_center_given_flanks": p_center_given_flanks,
        "p_center_unconditional": p_center_unconditional,
        **results,
    }