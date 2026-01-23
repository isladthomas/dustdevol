from dustdevol.evolve import evolve_sfr
from dustdevol.imf import chab
import dustdevol.generic as g
from dustdevol.DeVis2017 import (
    xSFR_inflow,
    xSFR_outflow,
    grain_growth,
    dust_destruction,
    stellar_ejecta,
    fast_ejecta_lin,
    fast_ejecta_log,
)
from numpy import arange, zeros
import matplotlib.pyplot as plt
from timeit import default_timer as timer

times = zeros(1)

for i in range(1):
    start = timer()
    results_slow = evolve_sfr(
        g.fp(0),
        g.fp(20),
        arange(0.05, 20, 0.05, dtype=g.fp),
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
    )
    end = timer()
    times[i] = end - start

print("Standard Ejecta Mean: {} seconds".format(times.mean()))
print("Standard Ejecta Std : {} seconds".format(times.std()))

for i in range(1):
    start = timer()
    results_lin = evolve_sfr(
        g.fp(0),
        g.fp(20),
        arange(0.05, 20, 0.05, dtype=g.fp),
        g.sfr_from_file,
        chab,
        xSFR_inflow,
        xSFR_outflow,
        g.off,
        grain_growth,
        dust_destruction,
        fast_ejecta_lin,
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
    times[i] = end - start

print("Fast Ejecta Mean: {} seconds".format(times.mean()))
print("Fast Ejecta Std : {} seconds".format(times.std()))

for i in range(1):
    start = timer()
    results_log = evolve_sfr(
        g.fp(0),
        g.fp(20),
        arange(0.05, 20, 0.05, dtype=g.fp),
        g.sfr_from_file,
        chab,
        xSFR_inflow,
        xSFR_outflow,
        g.off,
        grain_growth,
        dust_destruction,
        fast_ejecta_log,
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
    times[i] = end - start

print("Fast Ejecta Mean: {} seconds".format(times.mean()))
print("Fast Ejecta Std : {} seconds".format(times.std()))

plt.figure(1)
plt.plot(results_slow[:,0], results_slow[:,1])
plt.plot(results_lin[:,0], results_lin[:,1])
plt.plot(results_log[:,0], results_log[:,1])
plt.yscale("log")
plt.legend(["standard", "fast lin", "fast log"])
plt.savefig("gas.png")

plt.figure(2)
plt.plot(results_slow[:,0], results_slow[:,2])
plt.plot(results_lin[:,0], results_lin[:,2])
plt.plot(results_log[:,0], results_log[:,2])
plt.yscale("log")
plt.legend(["standard", "fast lin", "fast log"])
plt.savefig("stars.png")

plt.figure(3)
plt.plot(results_slow[:,0], results_slow[:,3])
plt.plot(results_lin[:,0], results_lin[:,3])
plt.plot(results_log[:,0], results_log[:,3])
plt.yscale("log")
plt.legend(["standard", "fast lin", "fast log"])
plt.savefig("metals.png")

plt.figure(4)
plt.plot(results_slow[:,0], results_slow[:,4])
plt.plot(results_lin[:,0], results_lin[:,4])
plt.plot(results_log[:,0], results_log[:,4])
plt.yscale("log")
plt.legend(["standard", "fast lin", "fast log"])
plt.savefig("oxygen.png")

plt.figure(5)
plt.plot(results_slow[:,0], results_slow[:,5])
plt.plot(results_lin[:,0], results_lin[:,5])
plt.plot(results_log[:,0], results_log[:,5])
plt.yscale("log")
plt.legend(["standard", "fast lin", "fast log"])
plt.savefig("dust.png")
