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

for i, item in enumerate(inits_original):
    
    start = timer()

    ch = ChemModel(**item)

    snrate = ch.supernova_rate()
    all_results = ch.gas_metal_dust_mass(snrate)

    end = timer()
    print("Model " + item["name"] + " finished w/ og code in " + str(end - start))

    params = {
        "time": all_results[:, 0],
        "mgas": all_results[:, 1],
        "mstars": all_results[:, 2],
        "metalmass": all_results[:, 3],
        "dustmass": all_results[:, 5],
        "oxygenmass": all_results[:, 13],
    }

    plt.figure(5*i)
    plt.plot(params["time"], params["mgas"])
    
    plt.figure(5*i + 1)
    plt.plot(params["time"], params["mstars"])

    plt.figure(5*i + 2)
    plt.plot(params["time"], params["metalmass"])

    plt.figure(5*i + 3)
    plt.plot(params["time"], params["oxygenmass"])

    plt.figure(5*i + 4)
    plt.plot(params["time"], params["dustmass"])

inits_log = [
    {
        "time_start": 0,
        "time_end": 20,
        "times": params["time"],
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
        "times": params["time"],
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
        "times": params["time"],
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
        "times": params["time"],
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
        "times": params["time"],
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
        "times": params["time"],
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
        "times": params["time"],
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

for i, item in enumerate(inits_log):
    
    start = timer()

    results = evolve_sfr(**item)

    end = timer()

    print("Model " + inits_original[i]["name"] + " finished w/ log code in " + str(end - start))

    plt.figure(5*i)
    plt.plot(results[:,0], results[:,1])
    plt.yscale("log")
    plt.legend(["original", "new"])
    plt.savefig(inits_original[i]["name"] + "_gas.png")

    plt.figure(5*i + 1)
    plt.plot(results[:,0], results[:,2])
    plt.yscale("log")
    plt.legend(["original", "new"])
    plt.savefig(inits_original[i]["name"] + "_stars.png")

    plt.figure(5*i + 2)
    plt.plot(results[:,0], results[:,3])
    plt.yscale("log")
    plt.legend(["original", "new"])
    plt.savefig(inits_original[i]["name"] + "_metals.png")
    
    plt.figure(5*i + 3)
    plt.plot(results[:,0], results[:,4])
    plt.yscale("log")
    plt.legend(["original", "new"])
    plt.savefig(inits_original[i]["name"] + "_oxygen.png")

    plt.figure(5*i + 4)
    plt.plot(results[:,0], results[:,5])
    plt.yscale("log")
    plt.legend(["original", "new"])
    plt.savefig(inits_original[i]["name"] + "_dust.png")
