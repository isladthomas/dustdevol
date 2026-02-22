from dustdevol.evolve import evolve_2o
from dustdevol.evolve_FCRK import evolve_2o_FC
from dustdevol.imf import chab
import dustdevol.generic as g
from dustdevol.DeVis2017 import (
    xSFR_inflow,
    xSFR_outflow,
    grain_growth,
    dust_destruction,
    stellar_ejecta,
)
import dustdevol.BEDE2 as bede
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt

start = timer()
results = evolve_2o(
    g.fp(0),
    g.fp(13.79),
    g.sfr_from_file,
    chab,
    xSFR_inflow,
    xSFR_outflow,
    g.off,
    grain_growth,
    dust_destruction,
    stellar_ejecta,
    [4e10],
    [0],
    [0, 0],
    [0],
    {
        "sfr_file": "Milkyway_2017.sfh",
        "sn_dust_reduction": 1,
        "sn_destruction": 0,
        "inflow_xSFR": 0,
        "outflow_xSFR": 0,
        "cold_fraction": 0.5,
        "grain_growth_epsilon": 0,
        "stellar_lifetimes": g.S92,
        "dust_yields": g.TF01,
        "metal_yields": g.vdHG97_M92_yields,
        "yield_table_z_cutoffs": g.vdHG97_M92_cutoffs,
    },
    1,
    1e-3,
)
end = timer()
time = end - start

print("Model I finished in {} seconds".format(time))
print("Only took {} steps!".format(len(results["times"])))
print("(out of {} attempted steps...)".format(
    results["cache"]["attempted_steps"]))
print("Smallest step was {} Gyr".format(np.diff(results["times"]).min()))

start = timer()
results_fast = evolve_2o_FC(
    g.fp(0),
    g.fp(13.79),
    g.sfr_from_file,
    chab,
    xSFR_inflow,
    xSFR_outflow,
    g.off,
    grain_growth,
    bede.dust_destruction,
    bede.stellar_ejecta,
    g.fp_array([4e10]),
    g.fp_array([0]),
    g.fp_array([0, 0]),
    g.fp_array([0]),
    {
        "sfr_file": "Milkyway_2017.sfh",
        "sn_dust_reduction": g.fp(1),
        "sn_destruction": g.fp(0),
        "inflow_xSFR": g.fp(0),
        "outflow_xSFR": g.fp(0),
        "cold_fraction": g.fp(0.5),
        "grain_growth_epsilon": g.fp(0),
        "stellar_lifetimes": g.stellar_lifetimes,
        "dust_yields": g.TF01,
        "metal_yields": g.vdHG97_M92_yields,
        "yield_table_z_cutoffs": g.vdHG97_M92_cutoffs,
    },
    g.fp(1),
    g.fp(1e-3),
)
end = timer()
time = end - start

print("Model I finished in {} seconds".format(time))
print("Only took {} steps!".format(len(results_fast["times"])))
print("(out of {} attempted steps...)".format(
    results_fast["cache"]["attempted_steps"]))
print("Smallest step was {} Gyr".format(np.diff(results_fast["times"]).min()))
start = timer()

plt.plot(results["times"], results["gas_masses"])
plt.plot(results_fast["times"], results_fast["gas_masses"])
plt.yscale("log")
plt.legend(["Adaptive Code", "Z at Birth Code"])
plt.savefig("gas_adaptive.png")
plt.clf()

plt.plot(results["times"], results["star_masses"])
plt.plot(results_fast["times"], results_fast["star_masses"])
plt.yscale("log")
plt.legend(["Adaptive Code", "Z at Birth Code"])
plt.savefig("stars_adaptive.png")
plt.clf()

plt.plot(results["times"], results["metal_masses"][:, 0])
plt.plot(results_fast["times"], results_fast["metal_masses"][:, 0])
plt.yscale("log")
plt.legend(["Adaptive Code", "Z at Birth Code"])
plt.savefig("metals_adaptive.png")
plt.clf()

plt.plot(results["times"], results["metal_masses"][:, 1])
plt.plot(results_fast["times"], results_fast["metal_masses"][:, 1])
plt.yscale("log")
plt.legend(["Adaptive Code", "Z at Birth Code"])
plt.savefig("oxygen_adaptive.png")
plt.clf()

plt.plot(results["times"], results["dust_masses"])
plt.plot(results_fast["times"], results_fast["dust_masses"])
plt.yscale("log")
plt.legend(["Adaptive Code", "Z at Birth Code"])
plt.savefig("dust_adaptive.png")
plt.clf()

end = 0.03
to_plot = results["times"] <= end
to_plot_fast = results_fast["times"] <= end

plt.plot(results["times"][to_plot], np.diff(results["times"])[to_plot[:-1]])
plt.plot(
    results_fast["times"][to_plot_fast],
    np.diff(results_fast["times"])[to_plot_fast[:-1]],
)
plt.xlim(0, end)
plt.yscale("log")
plt.legend(["Adaptive Code", "Z at Birth Code"])
plt.grid()
plt.savefig("timesteps.png")
plt.clf()
