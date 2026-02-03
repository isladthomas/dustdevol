from dustdevol.evolve import evolve_sfr
from dustdevol.imf import chab, bad_chab
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
from copy import deepcopy

inits_original = [
    {
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
    {
        "name": "Model_II_test",
        "gasmass_init": 4e10,
        "SFH": "delayed.sfh",
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
    {
        "name": "Model_III_test",
        "gasmass_init": 4e10,
        "SFH": "delayed.sfh",
        "t_end": 20.0,
        "gamma": 0,
        "IMF_fn": "Chab",
        "dust_source": "SN+LIMS",
        "reduce_sn_dust": False,
        "destroy": False,
        "inflows": {"metals": 0.0, "xSFR": 0, "dust": 0},
        "outflows": {"metals": True, "xSFR": 1.5, "dust": True},
        "cold_gas_fraction": 0.5,
        "epsilon_grain": 0,
        "destruct": 0,
    },
    {
        "name": "Model_IV_test",
        "gasmass_init": 4e10,
        "SFH": "delayed.sfh",
        "t_end": 20.0,
        "gamma": 0,
        "IMF_fn": "Chab",
        "dust_source": "All",
        "reduce_sn_dust": 6,
        "destroy": True,
        "inflows": {"metals": 0.0, "xSFR": 1.7, "dust": 0},
        "outflows": {"metals": True, "xSFR": 1.7, "dust": True},
        "cold_gas_fraction": 0.5,
        "epsilon_grain": 700,
        "destruct": 150,
    },
    {
        "name": "Model_V_test",
        "gasmass_init": 4e10,
        "SFH": "delayed.sfh",
        "t_end": 20.0,
        "gamma": 0,
        "IMF_fn": "Chab",
        "dust_source": "All",
        "reduce_sn_dust": 12,
        "destroy": True,
        "inflows": {"metals": 0.0, "xSFR": 2.5, "dust": 0},
        "outflows": {"metals": True, "xSFR": 2.5, "dust": True},
        "cold_gas_fraction": 0.5,
        "epsilon_grain": 5000,
        "destruct": 1500,
    },
    {
        "name": "Model_VI_test",  # needs to be run till 60Gyrs
        "gasmass_init": 4e10,
        "SFH": "delayed_over_3.sfh",
        "t_end": 20.0,
        "gamma": 0,
        "IMF_fn": "Chab",
        "dust_source": "All",
        "reduce_sn_dust": 100,
        "destroy": True,
        "inflows": {"metals": 0.0, "xSFR": 2.5, "dust": 0},
        "outflows": {"metals": True, "xSFR": 2.5, "dust": True},
        "cold_gas_fraction": 0.5,
        "epsilon_grain": 8000,
        "destruct": 150,
    },
    {
        "name": "Model_VII_test",
        "gasmass_init": 4e10,
        "SFH": "burst.sfh",
        "t_end": 20.0,
        "gamma": 0,
        "IMF_fn": "Chab",
        "dust_source": "All",
        "reduce_sn_dust": 12,
        "destroy": True,
        "inflows": {"metals": 0.0, "xSFR": 4, "dust": 0},
        "outflows": {"metals": True, "xSFR": 4, "dust": True},
        "cold_gas_fraction": 0.5,
        "epsilon_grain": 12000,
        "destruct": 150,
    },
]

standard_results = {}

for i, item in enumerate(inits_original):
    
    start = timer()

    ch = ChemModel(**item)

    snrate = ch.supernova_rate()
    all_results = ch.gas_metal_dust_mass(snrate)

    end = timer()
    print("Model " + item["name"] + " finished w/ og code in " + str(end - start))

    standard_results[item["name"]] = {
        "time": all_results[:, 0],
        "mgas": all_results[:, 1],
        "mstars": all_results[:, 2],
        "metalmass": all_results[:, 3],
        "dustmass": all_results[:, 5],
        "oxygenmass": all_results[:, 13],
    }

    plt.figure(5*i)
    plt.plot(standard_results[item["name"]]["time"], standard_results[item["name"]]["mgas"])
    
    plt.figure(5*i + 1)
    plt.plot(standard_results[item["name"]]["time"], standard_results[item["name"]]["mstars"])

    plt.figure(5*i + 2)
    plt.plot(standard_results[item["name"]]["time"], standard_results[item["name"]]["metalmass"])

    plt.figure(5*i + 3)
    plt.plot(standard_results[item["name"]]["time"], standard_results[item["name"]]["oxygenmass"])

    plt.figure(5*i + 4)
    plt.plot(standard_results[item["name"]]["time"], standard_results[item["name"]]["dustmass"])

inits_good = [
    {
        "time_start": 0,
        "time_end": 20,
        "times": standard_results[inits_original[0]["name"]]["time"],
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": dust_destruction,
        "ejecta_model": fast_ejecta_log,
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
    },
    {
        "time_start": 0,
        "time_end": 20,
        "times": standard_results[inits_original[1]["name"]]["time"],
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": dust_destruction,
        "ejecta_model": fast_ejecta_log,
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
    },
    {
        "time_start": 0,
        "time_end": 20,
        "times": standard_results[inits_original[2]["name"]]["time"],
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": dust_destruction,
        "ejecta_model": fast_ejecta_log,
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
    },
    {
        "time_start": 0,
        "time_end": 20,
        "times": standard_results[inits_original[3]["name"]]["time"],
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": dust_destruction,
        "ejecta_model": fast_ejecta_log,
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
    },
    {
        "time_start": 0,
        "time_end": 20,
        "times": standard_results[inits_original[4]["name"]]["time"],
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": dust_destruction,
        "ejecta_model": fast_ejecta_log,
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
    },
    {
        "time_start": 0,
        "time_end": 20,
        "times": standard_results[inits_original[5]["name"]]["time"],
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": dust_destruction,
        "ejecta_model": fast_ejecta_log,
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
    },
    {
        "time_start": 0,
        "time_end": 20,
        "times": standard_results[inits_original[6]["name"]]["time"],
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": dust_destruction,
        "ejecta_model": fast_ejecta_log,
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
    },
]

inits_bad = deepcopy(inits_good)

for item in inits_bad:
    item["imf"] = bad_chab

good_results = {}
bad_results = {}

for i, item in enumerate(zip(inits_good, inits_bad)):

    model_name = inits_original[i]["name"]
    
    start = timer()

    good_results[model_name] = evolve_sfr(**item[0])

    end = timer()

    print("Model " + model_name + " finished w/ new chab in " + str(end - start))

    start = timer()

    bad_results[model_name] = evolve_sfr(**item[1])

    end = timer()

    print("Model " + model_name + " finished w/ old chab in " + str(end - start))

    plt.figure(5*i)
    plt.plot(good_results[model_name][:,0], good_results[model_name][:,1])
    plt.plot(bad_results[model_name][:,0], bad_results[model_name][:,1])
    plt.suptitle(model_name)
    plt.title("Gas")
    plt.ylabel("Gas Mass (Msol)")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.legend(["Old Code", "New Code, Good Chab", "New Code, Bad Chab"])
    plt.savefig(model_name + "_gas.png")

    plt.clf()

    plt.plot(good_results[model_name][1:,0],
            (good_results[model_name][1:,1] -
             standard_results[model_name]["mgas"]) / 
             standard_results[model_name]["mgas"])
    plt.plot(bad_results[model_name][1:,0],
            (bad_results[model_name][1:,1] -
             standard_results[model_name]["mgas"]) / 
             standard_results[model_name]["mgas"])
    plt.suptitle(model_name)
    plt.title("Gas Error Relative to 2017 Code")
    plt.ylabel("Percent Error")
    plt.xlabel("Time (Gyr)")
    plt.ylim([-1,1])
    plt.legend(["Good Chab", "Bad Chab"])
    plt.savefig(model_name + "_gas_error.png")

    plt.close(5*i)

    plt.figure(5*i + 1)
    plt.plot(good_results[model_name][1:,0], good_results[model_name][1:,2])
    plt.plot(bad_results[model_name][1:,0], bad_results[model_name][1:,2])
    plt.suptitle(model_name)
    plt.title("Stars")
    plt.ylabel("Stellar Mass (Msol)")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.legend(["Old Code, Bad Chab", "New Code, Good Chab", "New Code, Bad Chab"])
    plt.savefig(model_name + "_stars.png")

    plt.clf()

    plt.plot(good_results[model_name][1:,0],
            (good_results[model_name][1:,2] -
             standard_results[model_name]["mstars"]) / 
             standard_results[model_name]["mstars"])
    plt.plot(bad_results[model_name][1:,0],
            (bad_results[model_name][1:,2] -
             standard_results[model_name]["mstars"]) / 
             standard_results[model_name]["mstars"])
    plt.suptitle(model_name)
    plt.title("Stars Error Relative to 2017 Code")
    plt.ylabel("Percent Error")
    plt.xlabel("Time (Gyr)")
    plt.ylim([-1,1])
    plt.legend(["Good Chab", "Bad Chab"])
    plt.savefig(model_name + "_stars_error.png")

    plt.close(5*i + 1)

    plt.figure(5*i + 2)
    plt.plot(good_results[model_name][1:,0], good_results[model_name][1:,3])
    plt.plot(bad_results[model_name][1:,0], bad_results[model_name][1:,3])
    plt.suptitle(model_name)
    plt.title("Metals")
    plt.ylabel("Metal Mass (Msol)")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.legend(["Old Code, Bad Chab", "New Code, Good Chab", "New Code, Bad Chab"])
    plt.savefig(model_name + "_metals.png")

    plt.clf()

    plt.plot(good_results[model_name][1:,0],
            (good_results[model_name][1:,3] -
             standard_results[model_name]["metalmass"]) / 
             standard_results[model_name]["metalmass"])
    plt.plot(bad_results[model_name][1:,0],
            (bad_results[model_name][1:,3] -
             standard_results[model_name]["metalmass"]) / 
             standard_results[model_name]["metalmass"])
    plt.suptitle(model_name)
    plt.title("Metals Error Relative to 2017 Code")
    plt.ylabel("Percent Error")
    plt.xlabel("Time (Gyr)")
    plt.ylim([-1,1])
    plt.legend(["Good Chab", "Bad Chab"])
    plt.savefig(model_name + "_metals_error.png")

    plt.close(5*i + 2)
    
    plt.figure(5*i + 3)
    plt.plot(good_results[model_name][1:,0], good_results[model_name][1:,4])
    plt.plot(bad_results[model_name][1:,0], bad_results[model_name][1:,4])
    plt.suptitle(model_name)
    plt.title("Oxygen")
    plt.ylabel("Oxygen Mass (Msol)")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.legend(["Old Code, Bad Chab", "New Code, Good Chab", "New Code, Bad Chab"])
    plt.savefig(model_name + "_oxygen.png")

    plt.clf()

    plt.plot(good_results[model_name][1:,0],
            (good_results[model_name][1:,4] -
             standard_results[model_name]["oxygenmass"]) / 
             standard_results[model_name]["oxygenmass"])
    plt.plot(bad_results[model_name][1:,0],
            (bad_results[model_name][1:,4] -
             standard_results[model_name]["oxygenmass"]) / 
             standard_results[model_name]["oxygenmass"])
    plt.suptitle(model_name)
    plt.title("Oxygen Error Relative to 2017 Code")
    plt.ylabel("Percent Error")
    plt.xlabel("Time (Gyr)")
    plt.ylim([-1,1])
    plt.legend(["Good Chab", "Bad Chab"])
    plt.savefig(model_name + "_oxygen_error.png")

    plt.close(5*i + 3)

    plt.figure(5*i + 4)
    plt.plot(good_results[model_name][1:,0], good_results[model_name][1:,5])
    plt.plot(bad_results[model_name][1:,0], bad_results[model_name][1:,5])
    plt.suptitle(model_name)
    plt.title("Dust")
    plt.ylabel("Dust Mass (Msol)")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.legend(["Old Code, Bad Chab", "New Code, Good Chab", "New Code, Bad Chab"])
    plt.savefig(model_name + "_dust.png")

    plt.clf()

    plt.plot(good_results[model_name][1:,0],
            (good_results[model_name][1:,5] -
             standard_results[model_name]["dustmass"]) / 
             standard_results[model_name]["dustmass"])
    plt.plot(bad_results[model_name][1:,0],
            (bad_results[model_name][1:,5] -
             standard_results[model_name]["dustmass"]) / 
             standard_results[model_name]["dustmass"])
    plt.suptitle(model_name)
    plt.title("Dust Error Relative to 2017 Code")
    plt.ylabel("Percent Error")
    plt.xlabel("Time (Gyr)")
    plt.ylim([-1,1])
    plt.legend(["Good Chab", "Bad Chab"])
    plt.savefig(model_name + "_dust_error.png")

    plt.close(5*i + 4)
