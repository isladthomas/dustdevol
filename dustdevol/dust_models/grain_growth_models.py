import dustdevol.generic as g
from numpy import minimum, maximum, inf


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
        model_params["grain_growth_epsilon_cloud"], mgas, sfr, mmetal[0], mdust
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
        model_params["grain_growth_epsilon_cloud"], mgas, sfr, mmetal[0], mdust
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
        model_params["grain_growth_epsilon_cloud"], mgas, sfr, mmetal[0], mdust
    )

    try:
        cache["gg_efficiency"][t] = (
            (
                ((1 - model_params["cold_fraction"]) / inf)
                + (model_params["cold_fraction"] / time_gg)
            )
            ** (-1)
        )[0]
        cache["gg_diffuse_timescale"][t] = inf
        cache["gg_cloud_timescale"][t] = time_gg[0]
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
                ((1 - model_params["cold_fraction"]) / inf)
                + (model_params["cold_fraction"] / time_gg)
            )
            ** (-1)
        )[0]
        cache["gg_diffuse_timescale"] = {}
        cache["gg_cloud_timescale"] = {}
        cache["gg_diffuse_timescale"][t] = inf
        cache["gg_cloud_timescale"][t] = time_gg[0]

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
        model_params["grain_growth_epsilon_diffuse"],
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
        model_params["grain_growth_epsilon_cloud"],
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
    t_grow = (0.0134 * (mgas**2)) / (sfr * e * mmetal * depletion * 0.1)

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
        model_params["grain_growth_epsilon_cloud"] * 3.731e-4, mgas, sfr, mmetal[0], mdust
    )

    try:
        cache["gg_efficiency"][t] = (
            (
                ((1 - model_params["cold_fraction"]) / inf)
                + (model_params["cold_fraction"] / time_gg)
            )
            ** (-1)
        )[0]
        cache["gg_diffuse_timescale"][t] = inf
        cache["gg_cloud_timescale"][t] = time_gg[0]
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
                ((1 - model_params["cold_fraction"]) / inf)
                + (model_params["cold_fraction"] / time_gg)
            )
            ** (-1)
        )[0]
        cache["gg_diffuse_timescale"] = {}
        cache["gg_cloud_timescale"] = {}
        cache["gg_diffuse_timescale"][t] = inf
        cache["gg_cloud_timescale"][t] = time_gg[0]

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
    """
    Normalizing to standard sfe (mstar = 1e9, mgas -> infty, z = 0),
    and ignoring depletion, this agrees with BEDE cloud presciption when
    e here is taken to be
    = e_BEDE * 3.731e-4 * SFE_0
    """
    t_grow = (0.00040 * mgas) / (mmetal * e)

    return t_grow
