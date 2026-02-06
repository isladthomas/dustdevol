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
import matplotlib.pyplot as plt
from numpy import diff
from timeit import default_timer as timer

inits_good = [
    {
        "time_start": 0,
        "time_end": 20,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": dust_destruction,
        "ejecta_model": fast_ejecta,
        "init_gas": [4e10],
        "init_star": [0],
        "init_metal": [0, 0],
        "init_dust": [0],
        "model_params": {
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
        "absolute_tolerance": 1,
        "relative_tolerance": 1e-3,
    },
    {
        "time_start": 0,
        "time_end": 20,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": dust_destruction,
        "ejecta_model": fast_ejecta,
        "init_gas": [4e10],
        "init_star": [0],
        "init_metal": [0, 0],
        "init_dust": [0],
        "model_params": {
            "sfr_file": "delayed.sfh",
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
        "absolute_tolerance": 1,
        "relative_tolerance": 1e-3,
    },
    {
        "time_start": 0,
        "time_end": 20,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": dust_destruction,
        "ejecta_model": fast_ejecta,
        "init_gas": [4e10],
        "init_star": [0],
        "init_metal": [0, 0],
        "init_dust": [0],
        "model_params": {
            "sfr_file": "delayed.sfh",
            "sn_dust_reduction": 1,
            "sn_destruction": 0,
            "inflow_xSFR": 0,
            "outflow_xSFR": 1.5,
            "outflow_metal": 1,
            "outflow_dust": 1,
            "cold_fraction": 0.5,
            "grain_growth_epsilon": 0,
            "stellar_lifetimes": g.S92,
            "dust_yields": g.TF01,
            "metal_yields": g.vdHG97_M92_yields,
            "yield_table_z_cutoffs": g.vdHG97_M92_cutoffs,
        },
        "absolute_tolerance": 1,
        "relative_tolerance": 1e-3,
    },
    {
        "time_start": 0,
        "time_end": 20,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": dust_destruction,
        "ejecta_model": fast_ejecta,
        "init_gas": [4e10],
        "init_star": [0],
        "init_metal": [0, 0],
        "init_dust": [0],
        "model_params": {
            "sfr_file": "delayed.sfh",
            "sn_dust_reduction": 6,
            "sn_destruction": 150,
            "inflow_xSFR": 1.7,
            "outflow_xSFR": 1.7,
            "outflow_metal": 1,
            "outflow_dust": 1,
            "cold_fraction": 0.5,
            "grain_growth_epsilon": 700,
            "stellar_lifetimes": g.S92,
            "dust_yields": g.TF01,
            "metal_yields": g.vdHG97_M92_yields,
            "yield_table_z_cutoffs": g.vdHG97_M92_cutoffs,
        },
        "absolute_tolerance": 1,
        "relative_tolerance": 1e-3,
    },
    {
        "time_start": 0,
        "time_end": 20,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": dust_destruction,
        "ejecta_model": fast_ejecta,
        "init_gas": [4e10],
        "init_star": [0],
        "init_metal": [0, 0],
        "init_dust": [0],
        "model_params": {
            "sfr_file": "delayed.sfh",
            "sn_dust_reduction": 12,
            "sn_destruction": 1500,
            "inflow_xSFR": 2.5,
            "outflow_xSFR": 2.5,
            "outflow_metal": 1,
            "outflow_dust": 1,
            "cold_fraction": 0.5,
            "grain_growth_epsilon": 5000,
            "stellar_lifetimes": g.S92,
            "dust_yields": g.TF01,
            "metal_yields": g.vdHG97_M92_yields,
            "yield_table_z_cutoffs": g.vdHG97_M92_cutoffs,
        },
        "absolute_tolerance": 1,
        "relative_tolerance": 1e-3,
    },
    {
        "time_start": 0,
        "time_end": 20,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": dust_destruction,
        "ejecta_model": fast_ejecta,
        "init_gas": [4e10],
        "init_star": [0],
        "init_metal": [0, 0],
        "init_dust": [0],
        "model_params": {
            "sfr_file": "delayed_over_3.sfh",
            "sn_dust_reduction": 100,
            "sn_destruction": 150,
            "inflow_xSFR": 2.5,
            "outflow_xSFR": 2.5,
            "outflow_metal": 1,
            "outflow_dust": 1,
            "cold_fraction": 0.5,
            "grain_growth_epsilon": 8000,
            "stellar_lifetimes": g.S92,
            "dust_yields": g.TF01,
            "metal_yields": g.vdHG97_M92_yields,
            "yield_table_z_cutoffs": g.vdHG97_M92_cutoffs,
        },
        "absolute_tolerance": 1,
        "relative_tolerance": 1e-3,
    },
    {
        "time_start": 0,
        "time_end": 20,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": dust_destruction,
        "ejecta_model": fast_ejecta,
        "init_gas": [4e10],
        "init_star": [0],
        "init_metal": [0, 0],
        "init_dust": [0],
        "model_params": {
            "sfr_file": "burst.sfh",
            "sn_dust_reduction": 12,
            "sn_destruction": 150,
            "inflow_xSFR": 4,
            "outflow_xSFR": 4,
            "outflow_metal": 1,
            "outflow_dust": 1,
            "cold_fraction": 0.5,
            "grain_growth_epsilon": 12000,
            "stellar_lifetimes": g.S92,
            "dust_yields": g.TF01,
            "metal_yields": g.vdHG97_M92_yields,
            "yield_table_z_cutoffs": g.vdHG97_M92_cutoffs,
        },
        "absolute_tolerance": 1,
        "relative_tolerance": 1e-3,
    },
]

good_results = {}
titles = ["I", "II", "III", "IV", "V", "VI", "VII"]

for i, item in enumerate(inits_good):

    model_name = titles[i]

    start = timer()

    good_results[model_name] = evolve_2o(**item)

    end = timer()

    print("Model " + model_name + " finished in " + str(end - start))
    print("Only took {} steps!".format(len(good_results[model_name]["times"])))
    print(
        "(out of {} attempted steps...)".format(
            good_results[model_name]["cache"]["attempted_steps"]
        )
    )
    print("Smallest step was {} seconds".format(
        diff(good_results[model_name]["times"]).min()))

    plt.figure(5 * i)
    plt.plot(good_results[model_name]["times"],
             good_results[model_name]["gas_masses"])
    plt.suptitle(model_name)
    plt.title("Gas")
    plt.ylabel("Gas Mass (Msol)")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.savefig(model_name + "_gas_adaptive.png")

    plt.close(5 * i)

    plt.figure(5 * i + 1)
    plt.plot(good_results[model_name]["times"],
             good_results[model_name]["star_masses"])
    plt.suptitle(model_name)
    plt.title("Stars")
    plt.ylabel("Stellar Mass (Msol)")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.savefig(model_name + "_stars_adaptive.png")

    plt.close(5 * i + 1)

    plt.figure(5 * i + 2)
    plt.plot(
        good_results[model_name]["times"],
        good_results[model_name]["metal_masses"][:, 0],
    )
    plt.suptitle(model_name)
    plt.title("Metals")
    plt.ylabel("Metal Mass (Msol)")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.savefig(model_name + "_metals_adaptive.png")

    plt.close(5 * i + 2)

    plt.figure(5 * i + 3)
    plt.plot(
        good_results[model_name]["times"],
        good_results[model_name]["metal_masses"][:, 1],
    )
    plt.suptitle(model_name)
    plt.title("Oxygen")
    plt.ylabel("Oxygen Mass (Msol)")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.savefig(model_name + "_oxygen_adaptive.png")

    plt.close(5 * i + 3)

    plt.figure(5 * i + 4)
    plt.plot(good_results[model_name]["times"],
             good_results[model_name]["dust_masses"])
    plt.suptitle(model_name)
    plt.title("Dust")
    plt.ylabel("Dust Mass (Msol)")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.savefig(model_name + "_dust_adaptive.png")

    plt.close(5 * i + 4)
