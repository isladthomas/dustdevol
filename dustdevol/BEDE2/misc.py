from numpy import where, logspace, log10, diff, clip
from dustdevol.generic import fp
from scipy.optimize import root


def life_from_mass_vec(masses, metal_hist, gas_hist, stellar_lifetimes, t, cache):
    """
    Function which finds stellar lifetime by solving the equation
    tau_f(Z(t - tau), m) - tau = 0
    for tau, taking into account the metallicity *at birth* for
    stars of mass m

    Parameters
    ----------
    masses : array_like
             list of masses to find lifetimes for
    metal_hist : function(float) -> (m,)
                 function which takes in a time and outputs the metal mass at
                 that time
    gas_hist : function(float) -> (g,)
               function which takes in a time and outputs the gas mass at
               that time
    stellar_lifetimes : function(ndarray, ndarray) -> ndarray
                        function which takes in an array of metallicities and
                        and array of initial masses, and outputs the lifetime
                        for that combination.
    t : float
        current time
    cache : dict
            dustdevol cache

    Returns
    -------
    out : ndarray
          Array of the same shape as `masses` with corresponding stellar
          lifetime.
    """

    try:
        tau0 = cache["sn_lifetimes"]
    except KeyError:
        cache["sn_lifetimes"] = stellar_lifetimes((fp(0), masses))
        tau0 = cache["sn_lifetimes"]

    if (
        abs(
            (
                stellar_lifetimes(
                    (
                        clip(
                            (metal_hist(t - tau0) / gas_hist(t - tau0))[:, 0],
                            fp(0.001),
                            fp(0.04),
                        ),
                        masses,
                    )
                )
                - tau0
            )
        ).max()
        > fp(2e-3)
    ):
        soln = root(
            lambda tau: stellar_lifetimes(
                (
                    clip((metal_hist(t - tau) / gas_hist(t - tau))
                         [:, 0], fp(0.001), fp(0.04)),
                    masses,
                )
            )
            - tau,
            tau0,
            method="krylov",
            options={"fatol": fp(2e-3)},
        ).x
    else:
        soln = tau0

    cache["sn_lifetimes"] = soln

    return soln


def supernova_rate(
    imf, metal_hist, gas_hist, sfr_hist, t, stellar_lifetimes, cache
):
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

    lifetimes = life_from_mass_vec(
        masses, metal_hist, gas_hist, stellar_lifetimes, t, cache
    )

    d_masses = where(t > lifetimes, d_masses, fp(0))

    sfr_vals = sfr_hist(t - lifetimes)

    sn_rate = (imf_vals * d_masses * sfr_vals).sum()

    return sn_rate
