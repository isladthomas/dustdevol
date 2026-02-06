from dustdevol.adaptive.evolve import evolve_2o
from dustdevol.adaptive.imf import chab
import dustdevol.adaptive.generic as g
from dustdevol.adaptive.DeVis2017 import (
    xSFR_inflow,
    xSFR_outflow,
    grain_growth,
    dust_destruction,
    fast_ejecta,
)
import dustdevol.evolve as e
import dustdevol.generic as g2
import dustdevol.DeVis2017 as D
import dustdevol.imf as i
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt

start = timer()
results = evolve_2o(
    g.fp(0),
    g.fp(20),
    g.sfr_from_file,
    chab,
    xSFR_inflow,
    xSFR_outflow,
    g.off,
    grain_growth,
    dust_destruction,
    fast_ejecta,
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
print("(out of {} attempted steps...)".format(results["cache"]["attempted_steps"]))
print("Smallest step was {} seconds".format(np.diff(results["times"]).min())) 

start = timer()
results_slow = e.evolve_sfr(
    g2.fp(0),
    g2.fp(20),
    np.append(np.linspace(0.001, 0.05, 1000, endpoint=False), np.arange(0.05, 20, 0.05)),
    g2.sfr_from_file,
    i.chab,
    D.xSFR_inflow,
    D.xSFR_outflow,
    g2.off,
    D.grain_growth,
    D.dust_destruction,
    D.fast_ejecta_log,
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
)
end = timer()
time = end - start

print("Model I finished in {} seconds".format(time))
print("This took {} steps.".format(len(results_slow[:, 0])))

plt.plot(results_slow[:, 0], results_slow[:, 1])
plt.plot(results["times"], results["gas_masses"])
plt.yscale("log")
plt.legend(["Original Code", "Adaptive Code"])
plt.savefig("gas_adaptive.png")
plt.clf()

plt.plot(results_slow[:, 0], results_slow[:, 2])
plt.plot(results["times"], results["star_masses"])
plt.yscale("log")
plt.legend(["Original Code", "Adaptive Code"])
plt.savefig("stars_adaptive.png")
plt.clf()

plt.plot(results_slow[:, 0], results_slow[:, 3])
plt.plot(results["times"], results["metal_masses"][:, 0])
plt.yscale("log")
plt.legend(["Original Code", "Adaptive Code"])
plt.savefig("metals_adaptive.png")
plt.clf()

plt.plot(results_slow[:, 0], results_slow[:, 4])
plt.plot(results["times"], results["metal_masses"][:, 1])
plt.yscale("log")
plt.legend(["Original Code", "Adaptive Code"])
plt.savefig("oxygen_adaptive.png")
plt.clf()

plt.plot(results_slow[:, 0], results_slow[:, 5])
plt.plot(results["times"], results["dust_masses"])
plt.yscale("log")
plt.legend(["Original Code", "Adaptive Code"])
plt.savefig("dust_adaptive.png")
plt.clf()

end = 0.1
to_plot = results["times"] <= end

plt.plot(results["times"][to_plot], np.diff(results["times"])[to_plot[:-1]])
plt.xlim(0, end)
plt.yscale("log")
plt.xticks(np.arange(0, end, 0.0028))
plt.grid()
plt.savefig("timesteps.png")
plt.clf()
