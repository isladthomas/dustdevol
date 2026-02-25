from numpy import maximum, minimum
from dustdevol.BEDE2 import supernova_rate
from dustdevol.generic import fp, fp_zeros


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
                   \"type_Ia_ratio\" : ratio of type Ia to type II SN

    Results
    -------
    output : (d,)
             float containing amount of dust destroyed (positive) in Msol/Gyr
    """

    sn_rate = supernova_rate(
        imf,
        metal_hist,
        gas_hist,
        sfr_hist,
        t,
        model_params["stellar_lifetimes"],
        model_params["type_Ia_ratio"],
        cache,
    )

    t_des = destruction_timescale(
        model_params["sn_destruction"], mgas, sn_rate)

    mdust_des = mdust * \
        (fp(1) - model_params["cold_fraction"]) * t_des ** fp(-1)

    return mdust_des


def THEMIS_dust_destruction(
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
    calculate dust destruction via supernova according to De Vis 2020,
    assuming that each sn event, on average, affects a specified amount
    of mass of the ISM, which is left as a free parameter, and that all dust
    in said mass is destroyed, with the additional stipulation (de Vis 2017)
    that destruction only occurs in the diffuse ISM.
    In addition, consider photofragmentation of non-silicate dust from UV
    photons from young, hot stars.

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
                   \"type_Ia_ratio\" : ratio of type Ia to type II SN
                   \"photofrag_efficiency\" : dimensionless parameter
                   controlling photofragmentation efficiency
                   \"silicate_fraction\" : fraction of dust made of silicates
                   and thus immune to photofragmentation

    Results
    -------
    output : (d,)
             float containing amount of dust destroyed (positive) in Msol/Gyr
    """

    sn_rate = supernova_rate(
        imf,
        metal_hist,
        gas_hist,
        sfr_hist,
        t,
        model_params["stellar_lifetimes"],
        model_params["type_Ia_ratio"],
        cache,
    )

    t_des = destruction_timescale(
        model_params["sn_destruction"], mgas, sn_rate)

    mdust_des = mdust * (1 - model_params["cold_fraction"]) * t_des**-1

    t_photo = photofragmentation_timescale(
        model_params["photofrag_efficiency"], sfr, mstar
    )

    mdust_frag = (
        mdust
        * (1 - model_params["cold_fraction"])
        * (1 - model_params["silicate_fraction"])
        * t_photo**-1
    )

    return mdust_des + mdust_frag


def THEMIS_grain_growth(
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
    Calculate dust grain growth according to the model in De Vis 2020
    assuming growth is proportional to the current dust mass and
    the inverse of a characteristic timescale (in Gyrs) that depends on
    the amount of *free* metals, and also assuming that only some metals
    are available for grain growth, with this fraction 2.45 higher in clouds
    than in the diffuse ISM, and also assuming that cloud GG is proportional
    to the SSFR, while diffuse GG is not.

    Parameters
    ----------
    model_params : dict
                   \"grain_growth_epsilon_diffuse\" : efficiency factor for
                   grain growth in diffuse ISM, should have same shape as
                   init_dust
                   \"grain_growth_epsilon_cloud\" : efficiency factor for
                   grain growth in molecular clouds, should have same shape as
                   init_dust
                   \"available_metals\": fraction of metals which can undergo
                   gg in the diffuse ISM (cloud is assumed 2.45x this)
                   \"cold_fraction\" : fraction of mass in molecular clouds
                   should have same shape as init_dust

    Returns
    -------
    out : (d,)
          dust grain growth in Msol/Gyr
    """
    # nab diffuse grain growth
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

    # turn any NaNs into zeros
    if any(diffuse_mdust_gg != diffuse_mdust_gg):
        diffuse_mdust_gg = fp_zeros(len(mdust))

    # nab cloud grain growth
    cloud_time_gg = cloud_BEDE_gt(
        model_params["grain_growth_epsilon_cloud"],
        mgas,
        sfr,
        mmetal[0],
        mdust,
        minimum(1.0, 2.45 * model_params["available_metals"]),
    )

    cloud_mdust_gg = mdust * model_params["cold_fraction"] * cloud_time_gg**-1

    # turn any NaNs into zeros
    if any(cloud_mdust_gg != cloud_mdust_gg):
        cloud_mdust_gg = fp_zeros(len(mdust))

    # add up grain growth from both sources. If it's somehow negative,
    # set it to zero
    mdust_gg = max(diffuse_mdust_gg + cloud_mdust_gg, 0)

    return mdust_gg


def diffuse_BEDE_gt(e, mgas, sfr, mmetal, mdust, available):
    """
    Characteristic timescale for GG in the diffuse ISM. Assumes gg is
    inversely proportional to an efficiency parameter, metallicity
    (relative to MW metallicity), and the depletion factor, i.e. how many
    metals (which can form dust) are already locked in dust.

    Parameters
    ----------
    e : float
        Grain growth efficiency parameter, units of per Gyr.
    mgas : ndarray, (g,)
           gas masses in Msol.
    sfr : ndarray, (s,)
          current sfr in Msol/Gyr.
    mmetal : float
             total metal mass in Msol.
    mdust : ndarray, (d,)
            dust masses in Msol.
    available : float
                fraction of metals that can possibly become dust through gg

    Results
    -------
    out : (d,)
          time in Gyr for 1 Msol of dust in a cloud to grow 1 Msol
    """
    depletion = maximum(0, 1 - (mdust / (mmetal * available)))
    t_grow = (0.0134 * mgas) / (e * mmetal * depletion)

    return t_grow


def cloud_BEDE_gt(e, mgas, sfr, mmetal, mdust, available):
    """
    Characteristic timescale for GG in a dense molecular cloud. Almost
    identical to ISM, except also proportional to SSFR, assuming that this
    tracks what portion of clouds are suitable for gg, and also divided by
    0.1, as it's assumed 90% of dust mass formed in clouds is loosely bound
    mantles that will be removed upon cloud dissociation.

    Parameters
    ----------
    e : float
        Grain growth efficiency parameter, units of per Gyr.
    mgas : ndarray, (g,)
           gas masses in Msol.
    sfr : ndarray, (s,)
          current sfr in Msol/Gyr.
    mmetal : float
             total metal mass in Msol.
    mdust : ndarray, (d,)
            dust masses in Msol.
    available : float
                fraction of metals that can possibly become dust through gg

    Results
    -------
    out : (d,)
          time in Gyr for 1 Msol of dust in a cloud to grow 1 Msol
    """
    depletion = maximum(0, 1 - (mdust / (mmetal * available)))
    t_grow = (0.0134 * (mgas**2)) / (sfr * e * mmetal * depletion * 0.1)

    return t_grow


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


def photofragmentation_timescale(efficiency, sfr, mstar):
    """
    Calculate timescale, in Gyrs, for 1 solar mass of gas to be swept up in
    a supernova.

    Parameters
    ----------
    efficiency : (d,)
                 dimensionless parameter representing photofrag efficiency
    sfr : float
          star formation rate in Msol / Gyr
    mstar : (s,)
           star mass in Msol

    Results
    -------
    out : (d,)
          Time in Gyr for 1 Msol of dust to be destroyed by photofrag
    """

    t_photo = mstar / (efficiency * sfr)

    return t_photo
