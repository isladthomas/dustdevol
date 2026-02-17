from dustdevol.adaptive.BEDE2 import fast_supernova_rate
from dustdevol.adaptive.generic import fp


def fast_dust_destruction(
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
    """
    calculate dust destruction via supernova according to Rowlands 2014,
    assuming that each sn event, on average, affects a specified amount
    of mass of the ISM, which is left as a free parameter, and that all dust
    in said mass is destroyed, with the additional stipulation (de Vis 2017)
    that destruction only occurs in the diffuse ISM
    requires:
        - \"stellar_lifetimes\": table where each row gives, in order
                                 the mass of a star in Msol, the lifetime
                                 of such a star in Gyrs in a low metallicity
                                 (Z < 0.008) environment, and the lifetime in
                                 a high metallicity (Z >= 0.008) environment
        - \"sn_destruction\": the amount of gas affected per supernova in Msol
        - \"cold_fraction\": fraction of mass in molecular clouds
                             should have same shape as init_dust
    """

    sn_rate = fast_supernova_rate(
        imf,
        metal_hist,
        gas_hist,
        sfr_hist,
        t,
        model_params["stellar_lifetimes"],
        mmetal[0] / mgas[0],
        cache,
    )

    t_des = destruction_timescale(
        model_params["sn_destruction"], mgas, sn_rate)

    mdust_des = mdust * (fp(1) - model_params["cold_fraction"]) * t_des**fp(-1)

    return mdust_des


def destruction_timescale(destruction, mgas, sn_rate):
    """
    calculate timescale, in Gyrs, for 1 solar mass of gas to be swept up in
    a supernova
    """

    t_destroy = mgas / (destruction * sn_rate)

    return t_destroy
