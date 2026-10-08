
#kinetics.py

def compute_spawn_rate(kon, TF_conc_current, N_A, empty_sites, c_on, num_bound,
                       empty_target=0, f=1.0):
    E_t = empty_target # empty target sites
    E_i = empty_sites - E_t # empty non-target sites

    Rspawn = (c_on**num_bound) * kon * TF_conc_current * N_A * (E_i + f * E_t)
    return Rspawn

def compute_off_rate(n_on_target, n_off_target, koff_target, koff_initial, num_bound, c_off):
    koff_t = koff_target /(c_off**num_bound)
    koff_i = koff_initial /(c_off**num_bound)
    Roff = n_on_target * koff_t +  n_off_target * koff_i

    slide_scale_target = koff_t / koff_i

    return Roff, koff_t, koff_i, slide_scale_target

def compute_slide_rate(n_on_target, n_off_target, hop_rate, koff_t, koff_i):
    Rslide = n_off_target * hop_rate + n_on_target * hop_rate * koff_t/koff_i
    return Rslide

