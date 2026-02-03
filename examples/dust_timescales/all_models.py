from dustdevol.evolve import evolve_sfr
from dustdevol.imf import chab, bad_chab
import dustdevol.generic as g
from dustdevol.DeVis2017 import (
    xSFR_inflow,
    xSFR_outflow,
    dust_destruction,
    stellar_ejecta,
    fast_ejecta_lin,
    fast_ejecta_log,
)
from chemevol import ChemModel
from numpy import arange, zeros, maximum, minimum, column_stack
import matplotlib.pyplot as plt
from timeit import default_timer as timer
from scipy import interpolate
from copy import deepcopy


def Mattson_gg(model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust):
    time_gg = Mattson_gt(
        model_params["grain_growth_epsilon"], mgas, sfr[i], mmetal[0], mdust
    )

    try:
        model_params["gg_timescale"][i] = (time_gg / (1.0 - (mdust / mmetal[0])))[0]
    except KeyError:
        model_params["gg_timescale"] = g.fp_zeros(len(times))
        model_params["gg_timescale"][i] = (time_gg / (1.0 - (mdust / mmetal[0])))[0]

    mdust_gg = mdust * (1.0 - (mdust / mmetal[0])) * time_gg**-1
    if any(mdust_gg != mdust_gg):
        mdust_gg = g.fp_zeros(len(mdust))

    return mdust_gg


def Mattson_gt(e, mgas, sfr, mmetal, mdust):
    t_grow = (mgas**2) / (e * mmetal * sfr)

    return t_grow


def bad_DeVis_gg(
    model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
):
    time_gg = bad_DeVis_gt(
        model_params["grain_growth_epsilon"], mgas, sfr[i], mmetal[0], mdust
    )

    try:
        model_params["gg_timescale"][i] = (time_gg / (
            (1.0 - (mdust / mmetal[0])) * model_params["cold_fraction"]
        ))[0]
    except KeyError:
        model_params["gg_timescale"] = g.fp_zeros(len(times))
        model_params["gg_timescale"][i] = (time_gg / (
            (1.0 - (mdust / mmetal[0])) * model_params["cold_fraction"]
        ))[0]

    mdust_gg = (
        mdust
        * model_params["cold_fraction"]
        * (1.0 - (mdust / mmetal[0]))
        * time_gg**-1
    )
    if any(mdust_gg != mdust_gg):
        mdust_gg = g.fp_zeros(len(mdust))

    return mdust_gg


def bad_DeVis_gt(e, mgas, sfr, mmetal, mdust):
    t_grow = (mgas**2) / (e * mmetal * sfr)
    t_grow = t_grow / (1.0 - (mdust / mmetal))

    return t_grow


def DeVis_gg(model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust):
    time_gg = DeVis_gt(
        model_params["grain_growth_epsilon"], mgas, sfr[i], mmetal[0], mdust
    )

    try:
        model_params["gg_timescale"][i] = (time_gg / (
            (1.0 - (mdust / mmetal[0])) * model_params["cold_fraction"]
        ))[0]
    except KeyError:
        model_params["gg_timescale"] = g.fp_zeros(len(times))
        model_params["gg_timescale"][i] = (time_gg / (
            (1.0 - (mdust / mmetal[0])) * model_params["cold_fraction"]
        ))[0]

    mdust_gg = (
        mdust
        * model_params["cold_fraction"]
        * (1.0 - (mdust / mmetal[0]))
        * time_gg**-1
    )
    if any(mdust_gg != mdust_gg):
        mdust_gg = g.fp_zeros(len(mdust))

    return mdust_gg


def DeVis_gt(e, mgas, sfr, mmetal, mdust):
    t_grow = (mgas**2) / (e * mmetal * sfr)
    t_grow = t_grow

    return t_grow


def BEDE_gg(model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust):
    diffuse_time_gg = diffuse_BEDE_gt(
        model_params["grain_growth_epsilon"] * (5.59 / 3820),
        mgas,
        sfr[i],
        mmetal[0],
        mdust,
        model_params["available_metals"],
    )

    diffuse_mdust_gg = mdust * \
        (1 - model_params["cold_fraction"]) * diffuse_time_gg**-1
    if any(diffuse_mdust_gg != diffuse_mdust_gg):
        diffuse_mdust_gg = g.fp_zeros(len(mdust))

    cloud_time_gg = cloud_BEDE_gt(
        model_params["grain_growth_epsilon"],
        mgas,
        sfr[i],
        mmetal[0],
        mdust,
        minimum(1.0, 2.45 * model_params["available_metals"]),
    )

    cloud_mdust_gg = mdust * model_params["cold_fraction"] * cloud_time_gg**-1
    if any(cloud_mdust_gg != cloud_mdust_gg):
        cloud_mdust_gg = g.fp_zeros(len(mdust))

    mdust_gg = diffuse_mdust_gg + cloud_mdust_gg

    try:
        model_params["gg_timescale"][i] = ((
            ((1 - model_params["cold_fraction"]) / diffuse_time_gg)
            + (model_params["cold_fraction"] / cloud_time_gg)
        ) ** (-1))[0]
    except KeyError:
        model_params["gg_timescale"] = g.fp_zeros(len(times))
        model_params["gg_timescale"][i] = ((
            ((1 - model_params["cold_fraction"]) / diffuse_time_gg)
            + (model_params["cold_fraction"] / cloud_time_gg)
        ) ** (-1))[0]

    return mdust_gg


def diffuse_BEDE_gt(e, mgas, sfr, mmetal, mdust, available):
    depletion = maximum(0, 1 - (mdust / (mmetal * available)))
    t_grow = (0.0134 * mgas) / (e * mmetal * depletion)

    return t_grow


def cloud_BEDE_gt(e, mgas, sfr, mmetal, mdust, available):
    depletion = maximum(0, 1 - (mdust / (mmetal * available)))
    t_grow = (0.1 * 0.0134 * (mgas**2)) / (sfr * e * mmetal * depletion)

    return t_grow


def Asano_gg(model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust):
    time_gg = Asano_gt(
        model_params["grain_growth_epsilon"], mgas, sfr[i], mmetal[0], mdust
    )

    try:
        model_params["gg_timescale"][i] = (time_gg / (
            (1.0 - (mdust / mmetal[0])) * model_params["cold_fraction"]
        ))[0]
    except KeyError:
        model_params["gg_timescale"] = g.fp_zeros(len(times))
        model_params["gg_timescale"][i] = (time_gg / (
            (1.0 - (mdust / mmetal[0])) * model_params["cold_fraction"]
        ))[0]

    mdust_gg = (
        mdust
        * model_params["cold_fraction"]
        * (1.0 - (mdust / mmetal[0]))
        * time_gg**-1
    )
    if any(mdust_gg != mdust_gg):
        mdust_gg = g.fp_zeros(len(mdust))

    return mdust_gg


def Asano_gt(e, mgas, sfr, mmetal, mdust):
    t_grow = (0.00040 * mgas) / mmetal

    return t_grow


inits_Mattson = [
    {
        "time_start": 0,
        "time_end": 20,
        "times": arange(0.05, 20, 0.05, dtype=g.fp),
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": Mattson_gg,
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
        "times": arange(0.05, 20, 0.05, dtype=g.fp),
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": Mattson_gg,
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
        "times": arange(0.05, 20, 0.05, dtype=g.fp),
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": Mattson_gg,
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
        "times": arange(0.05, 20, 0.05, dtype=g.fp),
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": Mattson_gg,
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
        "times": arange(0.05, 20, 0.05, dtype=g.fp),
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": Mattson_gg,
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
        "times": arange(0.05, 20, 0.05, dtype=g.fp),
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": Mattson_gg,
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
        "times": arange(0.05, 20, 0.05, dtype=g.fp),
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": Mattson_gg,
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

inits_bad_DeVis = deepcopy(inits_Mattson)
inits_DeVis = deepcopy(inits_Mattson)
inits_BEDE = deepcopy(inits_Mattson)
inits_Asano = deepcopy(inits_Mattson)

for item in inits_bad_DeVis:
    item["grain_growth_model"] = bad_DeVis_gg

for item in inits_DeVis:
    item["grain_growth_model"] = DeVis_gg

for item in inits_BEDE:
    item["grain_growth_model"] = BEDE_gg
    item["model_params"]["available_metals"] = 0.204

for item in inits_Asano:
    item["grain_growth_model"] = Asano_gg


inits = column_stack(
    (inits_Mattson, inits_bad_DeVis, inits_DeVis, inits_BEDE, inits_Asano)
)

titles = ["I", "II", "III", "IV", "V", "VI", "VII"]
legend = ["Mattson", "bad_DeVis", "DeVis", "BEDE", "Asano"]

for title, models in zip(titles, inits):

    for model in models:
        results = evolve_sfr(**model)
        plt.plot(results[:, 0], model["model_params"]["gg_timescale"])

    plt.suptitle("Model " + title)
    plt.title("Dust Growth Timescale")
    plt.ylabel("Timescale (Gyr Msol / Msol)")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.legend(legend)
    plt.savefig("Model_" + title + "_gg.png")
    plt.clf()
