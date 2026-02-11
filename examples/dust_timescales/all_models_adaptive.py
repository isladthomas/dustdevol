from dustdevol.adaptive.evolve import evolve_2o
from dustdevol.adaptive.imf import chab
import dustdevol.adaptive.generic as g
from dustdevol.adaptive.DeVis2017 import (
    xSFR_inflow,
    xSFR_outflow,
    dust_destruction,
    fast_ejecta,
)
from numpy import (
    maximum,
    minimum,
    column_stack,
    savez_compressed,
    array,
)
from copy import deepcopy
import logging

tol = 1e-6

logging.captureWarnings(True)
logging.basicConfig(filename="Warnings.log", level=logging.WARNING)


def Mattson_gg(
    model_params,
    sfr,
    imf,
    t,
    redshift,
    mgas,
    mstar,
    mmetal,
    mdust,
    gas_hist,
    star_hist,
    metal_hist,
    dust_hist,
    sfr_hist,
    cache,
):

    time_gg = Mattson_gt(
        model_params["grain_growth_epsilon"], mgas, sfr, mmetal[0], mdust
    )

    try:
        cache["gg_efficiency"][t] = (time_gg / (1.0 - (mdust / mmetal[0])))[0]
        cache["gg_timescale"][t] = time_gg[0]
        cache["gg_efficiency"] = {
            key: value for key, value in cache["gg_efficiency"].items() if key <= t
        }
        cache["gg_timescale"] = {
            key: value for key, value in cache["gg_timescale"].items() if key <= t
        }
    except KeyError:
        cache["gg_efficiency"] = {}
        cache["gg_efficiency"][t] = (time_gg / (1.0 - (mdust / mmetal[0])))[0]
        cache["gg_timescale"] = {}
        cache["gg_timescale"][t] = time_gg[0]

    mdust_gg = mdust * (1.0 - (mdust / mmetal[0])) * time_gg**-1
    if any(mdust_gg != mdust_gg):
        mdust_gg = g.fp_zeros(len(mdust))

    return mdust_gg


def Mattson_gt(e, mgas, sfr, mmetal, mdust):
    t_grow = (mgas**2) / (e * mmetal * sfr)

    return t_grow


def bad_DeVis_gg(
    model_params,
    sfr,
    imf,
    t,
    redshift,
    mgas,
    mstar,
    mmetal,
    mdust,
    gas_hist,
    star_hist,
    metal_hist,
    dust_hist,
    sfr_hist,
    cache,
):

    time_gg = bad_DeVis_gt(
        model_params["grain_growth_epsilon"], mgas, sfr, mmetal[0], mdust
    )

    try:
        cache["gg_efficiency"][t] = (
            time_gg / ((1.0 - (mdust / mmetal[0]))
                       * model_params["cold_fraction"])
        )[0]
        cache["gg_timescale"][t] = time_gg[0]
        cache["gg_efficiency"] = {
            key: value for key, value in cache["gg_efficiency"].items() if key <= t
        }
        cache["gg_timescale"] = {
            key: value for key, value in cache["gg_timescale"].items() if key <= t
        }
    except KeyError:
        cache["gg_efficiency"] = {}
        cache["gg_efficiency"][t] = (
            time_gg / ((1.0 - (mdust / mmetal[0]))
                       * model_params["cold_fraction"])
        )[0]
        cache["gg_timescale"] = {}
        cache["gg_timescale"][t] = time_gg[0]

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


def DeVis_gg(
    model_params,
    sfr,
    imf,
    t,
    redshift,
    mgas,
    mstar,
    mmetal,
    mdust,
    gas_hist,
    star_hist,
    metal_hist,
    dust_hist,
    sfr_hist,
    cache,
):

    time_gg = DeVis_gt(
        model_params["grain_growth_epsilon"], mgas, sfr, mmetal[0], mdust
    )

    try:
        cache["gg_efficiency"][t] = (
            time_gg / ((1.0 - (mdust / mmetal[0]))
                       * model_params["cold_fraction"])
        )[0]
        cache["gg_timescale"][t] = time_gg[0]
        cache["gg_efficiency"] = {
            key: value for key, value in cache["gg_efficiency"].items() if key <= t
        }
        cache["gg_timescale"] = {
            key: value for key, value in cache["gg_timescale"].items() if key <= t
        }
    except KeyError:
        cache["gg_efficiency"] = {}
        cache["gg_efficiency"][t] = (
            time_gg / ((1.0 - (mdust / mmetal[0]))
                       * model_params["cold_fraction"])
        )[0]
        cache["gg_timescale"] = {}
        cache["gg_timescale"][t] = time_gg[0]

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


def BEDE_gg(
    model_params,
    sfr,
    imf,
    t,
    redshift,
    mgas,
    mstar,
    mmetal,
    mdust,
    gas_hist,
    star_hist,
    metal_hist,
    dust_hist,
    sfr_hist,
    cache,
):

    diffuse_time_gg = diffuse_BEDE_gt(
        model_params["grain_growth_epsilon"] * (5.59 / 3820),
        mgas,
        sfr,
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
        sfr,
        mmetal[0],
        mdust,
        minimum(1.0, 2.45 * model_params["available_metals"]),
    )

    cloud_mdust_gg = mdust * model_params["cold_fraction"] * cloud_time_gg**-1
    if any(cloud_mdust_gg != cloud_mdust_gg):
        cloud_mdust_gg = g.fp_zeros(len(mdust))

    mdust_gg = max(diffuse_mdust_gg + cloud_mdust_gg, 0)

    try:
        cache["gg_efficiency"][t] = (
            (
                ((1 - model_params["cold_fraction"]) / diffuse_time_gg)
                + (model_params["cold_fraction"] / cloud_time_gg)
            )
            ** (-1)
        )[0]
        cache["gg_diffuse_timescale"][t] = diffuse_time_gg[0]
        cache["gg_cloud_timescale"][t] = cloud_time_gg[0]
        cache["gg_efficiency"] = {
            key: value for key, value in cache["gg_efficiency"].items() if key <= t
        }
        cache["gg_diffuse_timescale"] = {
            key: value
            for key, value in cache["gg_diffuse_timescale"].items()
            if key <= t
        }
        cache["gg_cloud_timescale"] = {
            key: value for key, value in cache["gg_cloud_timescale"].items() if key <= t
        }
    except KeyError:
        cache["gg_efficiency"] = {}
        cache["gg_efficiency"][t] = (
            (
                ((1 - model_params["cold_fraction"]) / diffuse_time_gg)
                + (model_params["cold_fraction"] / cloud_time_gg)
            )
            ** (-1)
        )[0]
        cache["gg_diffuse_timescale"] = {}
        cache["gg_cloud_timescale"] = {}
        cache["gg_diffuse_timescale"][t] = diffuse_time_gg[0]
        cache["gg_cloud_timescale"][t] = cloud_time_gg[0]

    return mdust_gg


def diffuse_BEDE_gt(e, mgas, sfr, mmetal, mdust, available):
    depletion = maximum(0, 1 - (mdust / (mmetal * available)))
    t_grow = (0.0134 * mgas) / (e * mmetal * depletion)

    return t_grow


def cloud_BEDE_gt(e, mgas, sfr, mmetal, mdust, available):
    depletion = maximum(0, 1 - (mdust / (mmetal * available)))
    t_grow = (0.1 * 0.0134 * (mgas**2)) / (sfr * e * mmetal * depletion)

    return t_grow


def Asano_gg(
    model_params,
    sfr,
    imf,
    t,
    redshift,
    mgas,
    mstar,
    mmetal,
    mdust,
    gas_hist,
    star_hist,
    metal_hist,
    dust_hist,
    sfr_hist,
    cache,
):

    time_gg = Asano_gt(
        model_params["grain_growth_epsilon"], mgas, sfr, mmetal[0], mdust
    )

    try:
        cache["gg_efficiency"][t] = (
            time_gg / ((1.0 - (mdust / mmetal[0]))
                       * model_params["cold_fraction"])
        )[0]
        cache["gg_timescale"][t] = time_gg[0]
        cache["gg_efficiency"] = {
            key: value for key, value in cache["gg_efficiency"].items() if key <= t
        }
        cache["gg_timescale"] = {
            key: value for key, value in cache["gg_timescale"].items() if key <= t
        }
    except KeyError:
        cache["gg_efficiency"] = {}
        cache["gg_efficiency"][t] = (
            time_gg / ((1.0 - (mdust / mmetal[0]))
                       * model_params["cold_fraction"])
        )[0]
        cache["gg_timescale"] = {}
        cache["gg_timescale"][t] = time_gg[0]

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
        "time_end": 13.79,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": Mattson_gg,
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
        "relative_tolerance": tol,
    },
    {
        "time_start": 0,
        "time_end": 13.79,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": Mattson_gg,
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
        "relative_tolerance": tol,
    },
    {
        "time_start": 0,
        "time_end": 13.79,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": Mattson_gg,
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
        "relative_tolerance": tol,
    },
    {
        "time_start": 0,
        "time_end": 13.79,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": Mattson_gg,
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
        "relative_tolerance": tol,
    },
    {
        "time_start": 0,
        "time_end": 13.79,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": Mattson_gg,
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
        "relative_tolerance": tol,
    },
    {
        "time_start": 0,
        "time_end": 13.79,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": Mattson_gg,
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
        "relative_tolerance": tol,
    },
    {
        "time_start": 0,
        "time_end": 13.79,
        "sfr_model": g.sfr_from_file,
        "imf": chab,
        "inflow_model": xSFR_inflow,
        "outflow_model": xSFR_outflow,
        "recycling_model": g.off,
        "grain_growth_model": Mattson_gg,
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
        "relative_tolerance": tol,
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

for title, models in zip(titles[3:], inits[3:]):

    for i, model in enumerate(models):
        print("Working on Model {} with {} gg".format(title, legend[i]))
        results = evolve_2o(**model)
        if legend[i] == "BEDE":
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
                "gg_efficiency": array(
                    [results["cache"]["gg_efficiency"][t]
                        for t in results["times"]]
                )[:, None],
                "gg_diffuse_timescale": array(
                    [
                        results["cache"]["gg_diffuse_timescale"][t]
                        for t in results["times"]
                    ]
                )[:, None],
                "gg_cloud_timescale": array(
                    [
                        results["cache"]["gg_cloud_timescale"][t]
                        for t in results["times"]
                    ]
                )[:, None],
            }
        else:
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
                "gg_efficiency": array(
                    [results["cache"]["gg_efficiency"][t]
                        for t in results["times"]]
                )[:, None],
                "gg_timescale": array(
                    [results["cache"]["gg_timescale"][t]
                        for t in results["times"]]
                )[:, None],
            }
        savez_compressed(
            "outputs/Model_{}_{}_gg".format(title, legend[i]), **output)
