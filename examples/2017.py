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
from chemevol import ChemModel
from numpy import arange, zeros
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from scipy import interpolate

n_trials = 10

times = zeros(n_trials)

for i in range(n_trials):
    ch = ChemModel(
        **{
            "name": "Model_I_test",
            "gasmass_init": 4e10,
            "SFH": "Milkyway_2017.sfh",
            "t_end": 20.0,
            "gamma": 0,
            "IMF_fn": "Chab",
            "dust_source": "SN+LIMS",
            "reduce_sn_dust": False,
            "destroy": False,
            "inflows": {"metals": 0.0, "xSFR": 0, "dust": 0},
            "outflows": {"metals": False, "xSFR": 0, "dust": False},
            "cold_gas_fraction": 0.5,
            "epsilon_grain": 0,
            "destruct": 0,
        },
    )
    start = timer()

    sn_rate = ch.supernova_rate()
    all_results = ch.gas_metal_dust_mass(sn_rate)

    end = timer()

    times[i] = end - start

print("Original Code Mean: {} seconds".format(times.mean()))
print("Original Code Std : {} seconds".format(times.std()))

params = {
    "time": all_results[:, 0],
    "mgas": all_results[:, 1],
    "mstars": all_results[:, 2],
    "metalmass": all_results[:, 3],
    "dustmass": all_results[:, 5],
    "oxygenmass": all_results[:, 13],
}

for i in range(n_trials):
    start = timer()
    results_slow = evolve_sfr(
        g.fp(0),
        g.fp(20),
        params["time"],
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

for i in range(n_trials):
    start = timer()
    results_lin = evolve_sfr(
        g.fp(0),
        g.fp(20),
        params["time"],
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

print("Lin Ejecta Mean: {} seconds".format(times.mean()))
print("Lin Ejecta Std : {} seconds".format(times.std()))

for i in range(n_trials):
    start = timer()
    results_log = evolve_sfr(
        g.fp(0),
        g.fp(20),
        params["time"],
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

print("Log Ejecta Mean: {} seconds".format(times.mean()))
print("Log Ejecta Std : {} seconds".format(times.std()))


original_gas = interpolate.make_interp_spline(params["time"], params["mgas"])
original_stars = interpolate.make_interp_spline(params["time"], params["mstars"])
original_metals = interpolate.make_interp_spline(params["time"], params["metalmass"])
original_oxygen = interpolate.make_interp_spline(params["time"], params["oxygenmass"])
original_dust = interpolate.make_interp_spline(params["time"], params["dustmass"])

plt.figure(1)
plt.plot(results_slow[:,0], results_slow[:,1])
plt.plot(results_lin[:,0], results_lin[:,1])
plt.plot(results_log[:,0], results_log[:,1])
plt.plot(params["time"], params["mgas"])
plt.yscale("log")
plt.legend(["standard", "fast lin", "fast log", "original code"])
plt.ylim(10 ** 9, 4.5 * 10 ** 10)
plt.savefig("gas_final.png")

plt.figure(2)
plt.plot(results_slow[:,0], results_slow[:,2])
plt.plot(results_lin[:,0], results_lin[:,2])
plt.plot(results_log[:,0], results_log[:,2])
plt.plot(params["time"], params["mstars"])
plt.yscale("log")
plt.legend(["standard", "fast lin", "fast log", "original code"])
plt.ylim(10 ** 8, 10 ** 11)
plt.savefig("stars_final.png")

plt.figure(3)
plt.plot(results_slow[:,0], results_slow[:,3])
plt.plot(results_lin[:,0], results_lin[:,3])
plt.plot(results_log[:,0], results_log[:,3])
plt.plot(params["time"], params["metalmass"])
plt.yscale("log")
plt.legend(["standard", "fast lin", "fast log", "original code"])
plt.ylim(10 ** 6, 10 ** 9)
plt.savefig("metals_final.png")

plt.figure(4)
plt.plot(results_slow[:,0], results_slow[:,4])
plt.plot(results_lin[:,0], results_lin[:,4])
plt.plot(results_log[:,0], results_log[:,4])
plt.plot(params["time"], params["oxygenmass"])
plt.yscale("log")
plt.legend(["standard", "fast lin", "fast log", "original code"])
plt.ylim(10 ** 5, 10 ** 9)
plt.savefig("oxygen_final.png")

plt.figure(5)
plt.plot(results_slow[:,0], results_slow[:,5])
plt.plot(results_lin[:,0], results_lin[:,5])
plt.plot(results_log[:,0], results_log[:,5])
plt.plot(params["time"], params["dustmass"])
plt.yscale("log")
plt.legend(["standard", "fast lin", "fast log", "original code"])
plt.ylim(10 ** 5, 10 ** 8)
plt.savefig("dust_final.png")
