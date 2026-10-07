import glob_plt
print(glob_plt.reload_metric_curve("kon", "target_occ"))


from promote_figure import promote_figure
import repwork_sweep as rs

promote_figure(
    name="fig02_target_occ_vs_kon",
    source_png="figures/repwork_Tmax4.9e4/target_occ_vs_kon.png",
    params={"run_name": rs.run_name, "base_config": rs.base_config,
            "sweeps": rs.sweeps, "n_replicates": rs.n_replicates},
    caption="Target occupancy vs kon, mean ± SEM over 10 replicates.",
)