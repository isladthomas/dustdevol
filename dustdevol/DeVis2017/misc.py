from numpy import where, logspace, log10, diff, searchsorted, clip
from dustdevol.adaptive.generic import fp, fp_zeros


def life_from_mass_vec(masses, stellar_lifetimes, metallicity):
    """
    Helper function which uses 0-order interpolation to find the lifetime
    in Gyr given a mass in Msol

    Parameters
    ----------
    masses : array_like
             list of masses to find lifetimes for
    stellar_lifetimes : 2D array
                        Array containing, in it's first column, a list of
                        star masses, second, their lifetimes at low
                        metallicities, and the third, lifetimes at high
                        metallicities.
    metallicity : str
                  str that reads "high" if high metallicity lifetimes are to
                  be used, and anything else if low values should be used.

    Returns
    -------
    out : ndarray
          Array of the same shape as `masses` with corresponding stellar
          lifetime.
    """

    """
    masses_half = stellar_lifetimes[:-1, 0] / \
        fp(2) + stellar_lifetimes[1:, 0] / fp(2)
    eff_indices = searchsorted(masses_half, masses)
    eff_indices = clip(eff_indices, 0, len(stellar_lifetimes[:, 0]) - 1)

    if metallicity == "high":
        return stellar_lifetimes[eff_indices, 2]
    else:
        return stellar_lifetimes[eff_indices, 1]
    """
    lifetimes = stellar_lifetimes(
        (
            clip(
                metallicity,
                fp(0.001),
                fp(0.04),
            ),
            masses,
        )
    )

    return lifetimes


def supernova_rate(imf, sfr_hist, t, stellar_lifetimes, metallicity_float, cache):
    """
    Calculate rate of supernova events in SN/Gyr, ignoring Type Ia SN using
    the "metallicity at death" approximation for finding lifetimes.

    Parameters
    ----------
    imf : function(ndarray) -> ndarray
          function which takes in an array of progenitor masses and spits out
          IMF values at that mass.
    sfr_hist : function(ndarray) -> ndarray
               function which takes in array of times and gives the sfr at
               that time
    t : float
        current time in the galacy
    stellar_lifetimes : 2D array
                        stellar lifetime table, fed to `life_from_mass_vec`
    metallicity_float : float
                        galaxy's current metallicity
    cache : dict
            cache for the dustdevol code

    Returns
    -------
    out : float
          supernova rate in SN/Gyr
    """

    try:
        masses = cache["sn_masses"]
        imf_vals = cache["sn_imf_values"]
        d_masses = cache["sn_d_masses"]
    except KeyError:
        masses = logspace(log10(8), log10(40), 257, dtype=fp)
        d_masses = diff(masses)
        masses = masses[:-1] + (d_masses / fp(2))
        imf_vals = imf(masses)

        cache["sn_masses"] = masses
        cache["sn_imf_values"] = imf_vals
        cache["sn_d_masses"] = d_masses

    if metallicity_float <= fp(0.008):
        metallicity = "low"

    else:
        metallicity = "high"

    lifetimes = life_from_mass_vec(masses, stellar_lifetimes, metallicity_float)

    d_masses = where(t > lifetimes, d_masses, fp(0))

    sfr_vals = sfr_hist(t - lifetimes)

    sn_rate = (imf_vals * d_masses * sfr_vals).sum()

    return sn_rate


def mass_from_life(t, stellar_lifetimes, metallicity):
    """
    Helper function which uses 0-order interpolation to find the
    mass in Msol of a star with (at most) a given lifetime in Gyr

    Paramters
    ---------
    t : float
        current galaxy lifetime
    stellar_lifetimes : 2D array
                        Array containing, in it's first column, a list of
                        star masses, second, their lifetimes at low
                        metallicities, and the third, lifetimes at high
                        metallicities.
    metallicity : float
                  current metallicity of galaxy
    """

    # find diff between requested lifetime and lifetime of each mass
    if metallicity == "high":
        diffs = t - stellar_lifetimes[:, 2]

    else:
        diffs = t - stellar_lifetimes[:, 1]

    # if the difference is negative, then the lifetime of such a star is
    # longer than the requested lifetime, so send those off, and *then* find
    # the closest value
    arg = where(diffs > fp(0), diffs, fp("inf")).argmin()

    return stellar_lifetimes[arg, 0]


def life_from_mass(m, stellar_lifetimes, metallicity):
    """
    Helper function which uses 0-order interpolation to find the lifetime
    in Gyr given a mass in Msol. Not vectorized, for multiple masses, or just
    in general, please use `life_from_mass_vec`

    Parameters
    ----------
    masses : float
             masses to find lifetime for
    stellar_lifetimes : 2D array
                        Array containing, in it's first column, a list of
                        star masses, second, their lifetimes at low
                        metallicities, and the third, lifetimes at high
                        metallicities.
    metallicity : str
                  str that reads "high" if high metallicity lifetimes are to
                  be used, and anything else if low values should be used.

    Returns
    -------
    out : float
          stellar lifetime for star of mass `m`
    """

    arg = (abs(stellar_lifetimes[:, 0] - m)).argmin()

    if metallicity == "high":
        return stellar_lifetimes[arg, 2]

    else:
        return stellar_lifetimes[arg, 1]


def xSFR_inflow(
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
    calculate gas inflow, and try to calculate metal and dust,
    if inflow metal and dust frac are not specified, assume 0

    Paramters
    ---------
    model_params : dict
        - \"inflow_xSFR\": multiple of SFR to calculate inflows
                           should have same shape as init_gas
    optional:
        - \"inflow_metal\": fraction of inflows in the form of metal
                            should have same shape as init_metal
        - \"inflow_dust\": fraction of inflows in the form of dust
                           should have same shape as init_dust

    Returns
    -------
    out : (g,), (m,), (d,)
          gas, metals, and dust gained from inflows in Msol/Gyr
    """

    gas_inflow = sfr * model_params["inflow_xSFR"]

    # if metal and dust frac not specified, set to zero
    # (so the error stops happening and we can go on quicker)
    try:
        metal_inflow = gas_inflow * model_params["inflow_metal"]
    except KeyError:
        model_params["inflow_metal"] = fp_zeros(len(mmetal))
        metal_inflow = fp_zeros(len(mmetal))

    try:
        dust_inflow = gas_inflow * model_params["inflow_dust"]
    except KeyError:
        model_params["inflow_dust"] = fp_zeros(len(mdust))
        dust_inflow = fp_zeros(len(mdust))

    return gas_inflow, metal_inflow, dust_inflow


def xSFR_outflow(
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
    Calculate gas inflow, and try to calculate metal and dust,
    if inflow metal and dust frac are not specified, assume 0

    Parameters
    ----------
    model_params : dict
        - \"inflow_xSFR\": multiple of SFR to calculate inflows
                           should have same shape as init_gas
    optional:
        - \"inflow_metal\": fraction of inflows in the form of metal
                            should have same shape as init_metal
        - \"inflow_dust\": fraction of inflows in the form of dust
                           should have same shape as init_dust

    Returns
    -------
    out : (g,), (m,), (d,)
          Gas, metal, and dust lost (positive) to outflows
    """

    gas_outflow = sfr * model_params["outflow_xSFR"]

    # if metal and dust frac not specified, set to zero
    # (so the error stops happening and we can go on quicker)
    try:
        metal_outflow = (mmetal / mgas[0]) * \
            gas_outflow * model_params["outflow_metal"]
    except KeyError:
        model_params["outflow_metal"] = fp_zeros(len(mmetal))
        metal_outflow = fp_zeros(len(mmetal))

    try:
        dust_outflow = (mdust / mgas[0]) * \
            gas_outflow * model_params["outflow_dust"]
    except KeyError:
        model_params["outflow_dust"] = fp_zeros(len(mdust))
        dust_outflow = fp_zeros(len(mdust))

    return gas_outflow, metal_outflow, dust_outflow
