from dustdevol.adaptive.generic import fp_zeros
from dustdevol.adaptive.DeVis2017 import supernova_rate


def grain_growth(
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
    calculate dust grain growth according to the model in Rowlands 2014
    assuming growth is proportional to the current dust mass and
    the inverse of a characteristic timescale (in Gyrs) that depends on
    the amount of *free* metals, with the additional
    modification (de Vis 2017) that grain growth only occurs in H2 clouds
    requires:
        - \"grain_growth_epsilon\": efficiency factor for grain growth
                                    should have same shape as init_dust
        - \"cold_fraction\": fraction of mass in molecular clouds
                             should have same shape as init_dust
    """

    # calculate growth timescale in Gyr
    time_gg = grow_timescale(
        model_params["grain_growth_epsilon"], mgas, sfr, mmetal[0], mdust
    )

    # rescale to account for metals already in dust and how much is in
    # molecular clouds to get grain growth in Msol / Gyr
    mdust_gg = (
        mdust
        * model_params["cold_fraction"]
        * (1.0 - (mdust / mmetal[0]))
        * time_gg**-1
    )

    # if 0/0 occurs, which only happens if there's no dust or metals,
    # mdust_gg will be NaN
    # check if that happened, and if it did, set mdust_gg to 0
    # (after all, no dust grain AND no metals kinda necessitates no gg)
    if any(mdust_gg != mdust_gg):
        mdust_gg = fp_zeros(len(mdust))

    return mdust_gg


def grow_timescale(e, mgas, sfr, mmetal, mdust):
    """
    calculate the characteristic timescale for grain growth, (Rowlands 2014)
    specifically the time (in Gyr) for 1 Msol of dust to accrete 1 Msol
    of metals. Assumes that the ratio sfr / mgas is a stand in for amount
    of mass in molecular clouds.
    """

    t_grow = (mgas**2) / (e * mmetal * sfr)

    return t_grow


def dust_destruction(
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

    sn_rate = supernova_rate(sfr, imf, sfr_hist, t, model_params["stellar_lifetimes"])

    t_des = destruction_timescale(model_params["sn_destruction"], mgas, sn_rate)

    mdust_des = mdust * (1 - model_params["cold_fraction"]) * t_des**-1

    return mdust_des


def destruction_timescale(destruction, mgas, sn_rate):
    """
    calculate timescale, in Gyrs, for 1 solar mass of gas to be swept up in
    a supernova
    """

    t_destroy = mgas / (destruction * sn_rate)

    return t_destroy
