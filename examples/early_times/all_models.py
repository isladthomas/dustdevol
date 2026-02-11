
from dustdevol.adaptive.evolve import evolve_2o
from dustdevol.adaptive.imf import chab
import dustdevol.adaptive.generic as g
from dustdevol.adaptive.DeVis2017 import (
    xSFR_inflow,
    xSFR_outflow,
    grain_growth,
    fast_dust_destruction,
    fast_ejecta,
)
from numpy import (
    column_stack,
    savez_compressed,
    array,
)
from copy import deepcopy
import logging

tol = 1

logging.captureWarnings(True)
logging.basicConfig(filename="Warnings.log", level=logging.WARNING)


inits_1 = [
    {
        "time_start": 0,
        "time_end": 0.03,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": fast_dust_destruction,
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
        "relative_tolerance": tol,
    },
    {
        "time_start": 0,
        "time_end": 0.03,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": fast_dust_destruction,
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
        "relative_tolerance": tol,
    },
    {
        "time_start": 0,
        "time_end": 0.03,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": fast_dust_destruction,
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
        "relative_tolerance": tol,
    },
    {
        "time_start": 0,
        "time_end": 0.03,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": fast_dust_destruction,
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
        "relative_tolerance": tol,
    },
    {
        "time_start": 0,
        "time_end": 0.03,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": fast_dust_destruction,
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
        "relative_tolerance": tol,
    },
    {
        "time_start": 0,
        "time_end": 0.03,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": fast_dust_destruction,
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
        "relative_tolerance": tol,
    },
    {
        "time_start": 0,
        "time_end": 0.03,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": grain_growth,
        "destruction_model": fast_dust_destruction,
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
        "relative_tolerance": tol,
    },
]

inits_1k = deepcopy(inits_1)
inits_1M = deepcopy(inits_1)
inits_1G = deepcopy(inits_1)
inits_1T = deepcopy(inits_1)
inits_1P = deepcopy(inits_1)
inits_1E = deepcopy(inits_1)

for item in inits_1k:
    item["relative_tolerance"] = 1e-3

for item in inits_1M:
    item["relative_tolerance"] = 1e-6

for item in inits_1G:
    item["relative_tolerance"] = 1e-9

for item in inits_1T:
    item["relative_tolerance"] = 1e-12

for item in inits_1P:
    item["relative_tolerance"] = 1e-15

for item in inits_1E:
    item["relative_tolerance"] = 1e-18

inits = column_stack(
    (inits_1, inits_1k, inits_1M, inits_1G, inits_1T, inits_1P, inits_1E)
)

titles = ["I", "II", "III", "IV", "V", "VI", "VII"]
legend = ["1e0", "1e-3", "1e-6", "1e-9", "1e-12", "1e-15", "1e-18"]

for title, models in zip(titles, inits):

    for i, model in enumerate(models):
        print("Working on Model {} with {} accuracy".format(title, legend[i]))
        results = evolve_2o(**model)
        output = {
            "times": results["times"],
            "gas_masses": results["gas_masses"],
            "star_masses": results["star_masses"],
            "metal_masses": results["metal_masses"],
            "dust_masses": results["dust_masses"],
            "dgas_masses": results["dgas_masses"],
            "dstar_masses": results["dstar_masses"],
            "dmetal_masses": results["dmetal_masses"],
            "ddust_masses": results["ddust_masses"],
            "sfr": results["sfr"],
        }
        savez_compressed(
            "outputs/Model_{}_{}_accuracy".format(title, legend[i]), **output)
