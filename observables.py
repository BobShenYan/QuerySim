#observables.py
#book keeping for simulation.py (taking the memory load)
import numpy as np

#.record method (active every iteration)

class Observer:
    def __init__(self, M, target_sites, emit_every, alpha=0.05):
        #methods won't need, but each instance/object will
        self.M = M 
        self.target_sites = target_sites
        self.emit_every = emit_every

        # the first emit starts as next emit
        self.next_emit_t = emit_every

        self.N_target = len(target_sites)

        # window accumulators (reset after each emit)
        # how long does the current state persist?

        # occupancy for avergaging?
        self.window_occ_accum = np.zeros(M, dtype=float)
        # occupnacy for n
        self.window_n_accum = 0.0
        # occupancy for num_bound
        self.window_num_bound = 0.0
        # time
        self.window_time = 0.0

        # weighted window tracking
        self.we_profile = np.zeros(M, dtype=float)
        # for smoothing (alpha is an arg, since its from user input)
        self.alpha = alpha

        # track through entire run, not reset
        self.profile_series = []
        self.profile_times = []

        self.occfrac_series = []
        self.occfrac_times = []

        self.occtarg_series = []
        self.occtarg_times = []

    # record method used for each iteration
    def record(self, occ, n, num_bound, dt, t):
        self.window_occ_accum += occ.astype(float) * dt
        self.window_n_accum += n * dt
        self.window_num_bound += num_bound * dt
        self.window_time += dt

        # t reached threshold, so time for record + emit
        if t >= self.next_emit_t:
            if self.window_time > 0:
                inst_profile = self.window_occ_accum/self.window_time
            else:
                inst_profile = np.zeros(self.M)

            #alpha for smoothing, remove if bothersome
            self.we_profile = (1 - self.alpha) * self.we_profile + self.alpha * inst_profile #new inst_profile has lower weight
            self.profile_series.append(self.we_profile.copy()) #so that independent snapshot we_profiles are stored in profile_series,not just the final we_profile
            self.profile_times.append(t)

            # --- occupancy by n ---
            if self.window_time > 0:
                n_avg = self.window_n_accum / self.window_time #time weighted n is averaged relative to window
                self.occfrac_series.append(n_avg/self.M)
                self.occfrac_times.append(t) #redundant to profile_times; occfrac is used when I only want smoothing for n/M plot

            # --- occupancy by num_bound ---
                num_avg = self.window_num_bound / self.window_time
                self.occtarg_series.append(num_avg/self.N_target)
                self.occtarg_times.append(t) #redudant but use specific

            self.window_occ_accum[:] = 0.0
            self.window_n_accum = 0.0  
            self.window_num_bound = 0.0
            self.window_time = 0.0
            self.next_emit_t += self.emit_every

    def get_results(self):
        return {
            "profile_series": np.array(self.profile_series),
            "profile_times": np.array(self.profile_times),
            
            #"target_occ_series": np.array(self.target_occ_series),
            "target_sites": self.target_sites,
            #"t_end": self.t,

            "occfrac_times": np.array(self.occfrac_times),
            "occfrac_series": np.array(self.occfrac_series),

            "occtarg_times": np.array(self.occtarg_times),
            "occtarg_series": np.array(self.occtarg_series),

            #"emit_events": np.array(self.emit_events),
        }




    
