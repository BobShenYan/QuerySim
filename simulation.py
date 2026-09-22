# simulation.py
from kinetics import compute_spawn_rate, compute_off_rate, compute_slide_rate
from observables import Observer

import numpy as np
import matplotlib.pyplot as plt
import random
#S(0) initialization
def run_OCC_SSA(
    M=200,
    target_sites=None,
    k_slide_eff= 1e1,          # bp^2/s taken from 1D diffusion coeff
    koff_initial=0.12,         # /s
    koff_target=0.01,          # /s
    kon=6.28e-21,              # /s -21
    TF_conc=1e-9,              # M
    N_A=6.022e26,               
    Tmax=10000.0,             # time limit (primary limit)
    max_events=int(1e10),       # event limit (2ndary limit)
    rng_seed=None,
    c_on=1.0,
    c_off=1.0,
    p_rebind=0.0,
    rebind_radius=5,
    rebind_tries=10,   
    emit_every=500.0  
    ): 
            
    t = 0.0

    #initialize rng keyword
    rng = np.random.default_rng(rng_seed)

    if target_sites is None:
        target_sites = np.linspace(M // 7, 6 * M // 7, 3, dtype=int)
    target_sites = np.unique(np.asarray(target_sites, dtype=int))
    is_target = np.zeros(M, dtype=bool)
    is_target[target_sites] = True

    tf_positions = np.empty(0, dtype=np.int64)
    occ = np.zeros(M, dtype=bool)
    target_count = 0
    n_tf_on_target = 0

    hop_rate = float(k_slide_eff)

    # observer initialized
    track1 = Observer(M, target_sites, emit_every)

    for ev in range(max_events):
        if t >= Tmax:
            break

        n = tf_positions.size
        num_bound = target_count

        empty_sites = M - n

        n_on_target = n_tf_on_target
        n_off_target = n - n_on_target

        Rspawn = compute_spawn_rate(kon, TF_conc, N_A, empty_sites, c_on, num_bound)
        Roff, koff_t, koff_i, slide_scale_target = compute_off_rate(n_on_target, n_off_target, koff_target, koff_initial, num_bound, c_off)
        Rslide = compute_slide_rate(n_on_target, n_off_target, hop_rate, koff_t, koff_i)

        R_tot = Rspawn + Rslide + Roff

        dt = rng.exponential(1/R_tot)

        t_old = t # is t_old only for pbar updating?
        t += dt

        track1.record(occ, n, num_bound, dt, t)

        u = rng.random() * R_tot
        if u < Rspawn:
            while True:
                pos = rng.integers(0, M)
                if not occ[pos]:
                    break
            tf_positions = np.append(tf_positions, pos)
            occ[pos] = True
            if is_target[pos]:
                target_count += 1
                n_tf_on_target += 1

        # ---- OFF ----
        elif u < Rspawn + Roff:
            if n == 0:
                continue

            #bool check of tf_positions
            on_t = is_target[tf_positions]
            weights = np.where(on_t, koff_t, koff_i)
            wsum = weights.sum()
            if wsum <= 0:
                continue

            idx = rng.choice(n, p=weights / wsum)
            old_pos = tf_positions[idx]

            tf_positions[idx] = tf_positions[-1]
            tf_positions = tf_positions[:-1]
            occ[old_pos] = False

            if is_target[old_pos]:
                target_count -= 1
                n_tf_on_target -= 1

            if rng.random() < p_rebind and tf_positions.size < M:
                for _ in range(rebind_tries):
                    cand = (old_pos + rng.integers(-rebind_radius, rebind_radius + 1)) % M
                    if not occ[cand]:
                        tf_positions = np.append(tf_positions, cand)
                        occ[cand] = True
                        if is_target[cand]:
                            target_count += 1
                            n_tf_on_target += 1
                        break

        # ---- SLIDE ----
        else:
            if n == 0:
                continue

            on_t = is_target[tf_positions]
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
            occ[cand] = True

            if is_target[cand]:
                target_count += 1
                n_tf_on_target += 1

    # we can get the results after the loop has finished
    results = track1.get_results()

    # choosing to return so that it's easier to work with the dict from .get_results()
    return {
        "end_t": t, 
        "tf_positions": tf_positions, 
        "occ": occ, 
        "target_count": target_count, 
        "n_tf_on_target": n_tf_on_target, 
        "events": ev + 1,
        **results,
    }
