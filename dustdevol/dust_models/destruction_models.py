from dustdevol.BEDE2 import supernova_rate


def destruction_timescale(destruction, mgas, sn_rate):

    t_destroy = mgas / (destruction * sn_rate)

    return t_destroy


def photofrag_timescale(mstar, sfr, efficiency):

    t_photofrag = mstar / (sfr * efficiency)

    return t_photofrag


def DeVis_dd(
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

    sn_rate = (
        supernova_rate(
            imf,
            metal_hist,
            gas_hist,
            sfr_hist,
            t,
            model_params["stellar_lifetimes"],
            cache,
        )
        + cache["Ia_rate"]
    )

    t_des = destruction_timescale(
        model_params["sn_destruction"], mgas, sn_rate)

    t_frag = photofrag_timescale(
        mstar, sfr, model_params["photofrag_efficiency"])

    try:
        cache["dd_efficiency"][t] = (
            (
                (
                    (1 - model_params["cold_fraction"])
                    * (1 - model_params["silicate_fraction"])
                    / t_frag
                )
                + ((1 - model_params["cold_fraction"]) / t_des)
            )
            ** (-1)
        )[0]
        cache["dd_frag_timescale"][t] = t_frag[0]
        cache["dd_des_timescale"][t] = t_des[0]
        cache["dd_efficiency"] = {
            key: value for key, value in cache["dd_efficiency"].items() if key <= t
        }
        cache["dd_frag_timescale"] = {
            key: value for key, value in cache["dd_frag_timescale"].items() if key <= t
        }
        cache["dd_des_timescale"] = {
            key: value for key, value in cache["dd_des_timescale"].items() if key <= t
        }
    except KeyError:
        cache["dd_efficiency"] = {}
        cache["dd_efficiency"][t] = (
            (
                (
                    (1 - model_params["cold_fraction"])
                    * (1 - model_params["silicate_fraction"])
                    / t_frag
                )
                + ((1 - model_params["cold_fraction"]) / t_des)
            )
            ** (-1)
        )[0]
        cache["dd_frag_timescale"] = {}
        cache["dd_des_timescale"] = {}
        cache["dd_frag_timescale"][t] = t_frag[0]
        cache["dd_des_timescale"][t] = t_des[0]

    mdust_des = mdust * (1 - model_params["cold_fraction"]) * t_des**-1
    mdust_frag = (
        mdust
        * (1 - model_params["cold_fraction"])
        * (1 - model_params["silicate_fraction"])
        * t_frag**-1
    )
    mdust_des[mdust_des != mdust_des] = 0
    mdust_frag[mdust_frag != mdust_frag] = 0

    return mdust_des + mdust_frag


def Asano_dd(
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

    sn_rate = (
        supernova_rate(
            imf,
            metal_hist,
            gas_hist,
            sfr_hist,
            t,
            model_params["stellar_lifetimes"],
            cache,
        )
        + cache["Ia_rate"]
    )

    t_des = destruction_timescale(
        (model_params["sn_destruction"] / (1.039**-0.289))
        * ((mmetal / mgas)[0] / 0.0134 + 0.039) ** -0.289,
        mgas,
        sn_rate,
    )

    t_frag = photofrag_timescale(
        mstar, sfr, model_params["photofrag_efficiency"])
    try:
        cache["dd_efficiency"][t] = (
            (
                (
                    (1 - model_params["cold_fraction"])
                    * (1 - model_params["silicate_fraction"])
                    / t_frag
                )
                + ((1 - model_params["cold_fraction"]) / t_des)
            )
            ** (-1)
        )[0]
        cache["dd_frag_timescale"][t] = t_frag[0]
        cache["dd_des_timescale"][t] = t_des[0]
        cache["dd_efficiency"] = {
            key: value for key, value in cache["dd_efficiency"].items() if key <= t
        }
        cache["dd_frag_timescale"] = {
            key: value for key, value in cache["dd_frag_timescale"].items() if key <= t
        }
        cache["dd_des_timescale"] = {
            key: value for key, value in cache["dd_des_timescale"].items() if key <= t
        }
    except KeyError:
        cache["dd_efficiency"] = {}
        cache["dd_efficiency"][t] = (
            (
                (
                    (1 - model_params["cold_fraction"])
                    * (1 - model_params["silicate_fraction"])
                    / t_frag
                )
                + ((1 - model_params["cold_fraction"]) / t_des)
            )
            ** (-1)
        )[0]
        cache["dd_frag_timescale"] = {}
        cache["dd_des_timescale"] = {}
        cache["dd_frag_timescale"][t] = t_frag[0]
        cache["dd_des_timescale"][t] = t_des[0]

    mdust_des = mdust * (1 - model_params["cold_fraction"]) * t_des**-1
    mdust_frag = (
        mdust
        * (1 - model_params["cold_fraction"])
        * (1 - model_params["silicate_fraction"])
        * t_frag**-1
    )
    mdust_des[mdust_des != mdust_des] = 0
    mdust_frag[mdust_frag != mdust_frag] = 0

    return mdust_des + mdust_frag


def Priestley_dd(
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

    sn_rate = (
        supernova_rate(
            imf,
            metal_hist,
            gas_hist,
            sfr_hist,
            t,
            model_params["stellar_lifetimes"],
            cache,
        )
        + cache["Ia_rate"]
    )

    t_des = destruction_timescale(
        (
            model_params["sn_destruction"]
            * (1 + (1 / 0.14))
            / (1 + ((mmetal / mgas)[0] / 0.001876))
        ),
        mgas,
        sn_rate,
    )

    t_frag = photofrag_timescale(
        mstar, sfr, model_params["photofrag_efficiency"])

    try:
        cache["dd_efficiency"][t] = (
            (
                (
                    (1 - model_params["cold_fraction"])
                    * (1 - model_params["silicate_fraction"])
                    / t_frag
                )
                + ((1 - model_params["cold_fraction"]) / t_des)
            )
            ** (-1)
        )[0]
        cache["dd_frag_timescale"][t] = t_frag[0]
        cache["dd_des_timescale"][t] = t_des[0]
        cache["dd_efficiency"] = {
            key: value for key, value in cache["dd_efficiency"].items() if key <= t
        }
        cache["dd_frag_timescale"] = {
            key: value for key, value in cache["dd_frag_timescale"].items() if key <= t
        }
        cache["dd_des_timescale"] = {
            key: value for key, value in cache["dd_des_timescale"].items() if key <= t
        }
    except KeyError:
        cache["dd_efficiency"] = {}
        cache["dd_efficiency"][t] = (
            (
                (
                    (1 - model_params["cold_fraction"])
                    * (1 - model_params["silicate_fraction"])
                    / t_frag
                )
                + ((1 - model_params["cold_fraction"]) / t_des)
            )
            ** (-1)
        )[0]
        cache["dd_frag_timescale"] = {}
        cache["dd_des_timescale"] = {}
        cache["dd_frag_timescale"][t] = t_frag[0]
        cache["dd_des_timescale"][t] = t_des[0]

    mdust_des = mdust * (1 - model_params["cold_fraction"]) * t_des**-1
    mdust_frag = (
        mdust
        * (1 - model_params["cold_fraction"])
        * (1 - model_params["silicate_fraction"])
        * t_frag**-1
    )
    mdust_des[mdust_des != mdust_des] = 0
    mdust_frag[mdust_frag != mdust_frag] = 0

    return mdust_des + mdust_frag
