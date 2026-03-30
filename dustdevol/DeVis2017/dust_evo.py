from numpy import where
from dustdevol.generic import fp, fp_zeros
from dustdevol.DeVis2017 import supernova_rate


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
    Calculate dust grain growth according to the model in Rowlands 2014
    assuming growth is proportional to the current dust mass and
    the inverse of a characteristic timescale (in Gyrs) that depends on
    the amount of *free* metals, with the additional
    modification (de Vis 2017) that grain growth only occurs in H2 clouds.

    Parameters
    ----------
    model_params : dict
                   \"grain_growth_epsilon\" : efficiency factor for grain
                   growth should have same shape as init_dust
                   \"cold_fraction\" : fraction of mass in molecular clouds
                   should have same shape as init_dust

    Returns
    -------
    out : (d,)
          dust grain growth in Msol/Gyr
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
        * (fp(1) - (mdust / mmetal[0]))
        * time_gg ** fp(-1)
    )

    # if 0/0 occurs, which only happens if there's no dust or metals,
    # mdust_gg will be NaN
    # check if that happened, and if it did, set mdust_gg to 0
    # (after all, no dust grain AND no metals kinda necessitates no gg)
    mdust_gg = where(mdust_gg == mdust_gg, mdust_gg, fp_zeros(len(mdust)))

    return mdust_gg


def grow_timescale(e, mgas, sfr, mmetal, mdust):
    """
    Calculate the characteristic timescale for grain growth, (Rowlands 2014)
    specifically the time (in Gyr) for 1 Msol of dust to accrete 1 Msol
    of metals. Assumes that the ratio sfr / mgas is a stand in for amount
    of mass in molecular clouds *that's undergoing significant grain growth*.

    Parameters
    ----------
    e : float
        Grain growth efficiency parameter, unitless.
    mgas : ndarray, (g,)
           gas masses in Msol.
    sfr : ndarray, (s,)
          current sfr in Msol/Gyr.
    mmetal : ndarray, (m,)
             metal masses in Msol.
    mdust : ndarray, (d,)
            dust masses in Msol.

    Results
    -------
    out : (d,)
          time in Gyr for 1 Msol of dust in a cloud to grow 1 Msol
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

    Parameters
    ----------
    model_params : dict
                   \"stellar_lifetimes\" : table where each row gives, in order
                   the mass of a star in Msol, the lifetime
                   of such a star in Gyrs in a low metallicity
                   (Z < 0.008) environment, and the lifetime in
                   a high metallicity (Z >= 0.008) environment
                   \"sn_destruction\" : amount of gas affected per SN in Msol
                   \"cold_fraction\" : fraction of mass in molecular clouds
                   should have same shape as init_dust

    Results
    -------
    output : (d,)
             float containing amount of dust destroyed (positive) in Msol/Gyr
    """

    sn_rate = supernova_rate(
        imf, sfr_hist, t, model_params["stellar_lifetimes"], mmetal[0] /
        mgas[0], cache
    ) + cache["Ia_rate"]

    t_des = destruction_timescale(
        model_params["sn_destruction"], mgas, sn_rate)

    mdust_des = mdust * \
        (fp(1) - model_params["cold_fraction"]) * t_des ** fp(-1)

    return mdust_des


def destruction_timescale(destruction, mgas, sn_rate):
    """
    Calculate timescale, in Gyrs, for 1 solar mass of gas to be swept up in
    a supernova.

    Parameters
    ----------
    destruction : (d,)
                  amount of gas swept up in a single SNe in Msol
    mgas : (g,)
           gas mass in Msol
    sn_rate : float
              number of SN in window [t, t+dt] (differential sense, nothing to
              do with stepsize of the integrator) in Number/Gyr

    Results
    -------
    out : (d,)
          Time in Gyr for 1 Msol of dust to be destroyed by SN
    """

    t_destroy = mgas / (destruction * sn_rate)

    return t_destroy
