from numpy import where, logspace, log10, diff, clip, exp, array, sort
from dustdevol.generic import fp
from scipy.optimize import root
from scipy.stats import poisson, uniform, loguniform


def sfr_from_efficiency(
    model_params,
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
    cache,
):
    """Formula for star formation rate according to De Vis 2020. Assumes
    a fixed reference star formation efficiency (sfr / mgas), and modulates
    it so that high stellar mass increases it, high redshift decreases it,
    and small gas fraction also decreases it.

    Parameters
    ----------
    model_params : dict
                   \"star_formation_efficiency\" : reference SFE, 8x the SFE
                   of the galaxy when Mstar is 1e9, gas fraction is 1, and
                   redshift is 0.

    Returns
    -------
    out : float
          sfr in Msol/Gyr.
    """

    sfe = (
        model_params["star_formation_efficiency"]
        * (mstar / 1e9) ** 0.25
        * (1 + exp(mstar / (10 * mgas))) ** -3
        * (1 + redshift) ** -1
    )

    sfr = sfe * mgas

    return sfr


def bursty_sfr_from_efficiency(
    model_params,
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
    cache,
):
    """Formula for star formation rate according to De Vis 2020. Assumes
    a fixed reference star formation efficiency (sfr / mgas), and modulates
    it so that high stellar mass increases it, high redshift decreases it,
    and small gas fraction also decreases it.
    Additionally, superimpose starbursts. The total number of bursts is
    poisson distributed assuming an average of 3.4475 occur during the age of
    the universe (0.5 every 2 Gyr over 13.79 Gyr), which are then made to
    start between 0 and 13.79 Gyr, distibuted uniformly, with lengths
    distributed uniformly between 0.03 and 0.3 Gyr, and with each producing
    a total stellar mass equal to between 0.004 and 0.4 times the total stellar
    mass at the start of the burst, with the multiplier distributed lognormally

    Parameters
    ----------
    model_params : dict
                   \"star_formation_efficiency\" : reference SFE, 8x the SFE
                   of the galaxy when Mstar is 1e9, gas fraction is 1, and
                   redshift is 0.

    Returns
    -------
    out : float
          sfr in Msol/Gyr.
    """

    try:
        burst_starts = cache["burst_starts"]
        burst_ends = cache["burst_ends"]
        burst_sizes = cache["burst_starts"]
    except KeyError:
        # calculate how many bursts we'll have
        burst_num = poisson.rvs(0.5 * (13.79 / 2))

        if burst_num != 0:

            # calculate the starts and ends of each burst. If any bursts
            # overlap, discard and retry.
            while True:
                burst_starts = sort(uniform.rvs(scale=13.79, size=burst_num))
                burst_lengths = uniform.rvs(
                    loc=0.03, scale=0.27, size=burst_num)
                burst_ends = burst_starts + burst_lengths
                if all(burst_starts[1:] > burst_ends[:-1]):
                    break

            # calculate the fraction of mstar generated in each burst,
            # distributed over the length of the burst
            burst_scales = loguniform.rvs(0.004, 0.4, size=burst_num)
            burst_sizes = burst_scales / burst_lengths

        else:
            burst_starts = array([])
            burst_ends = array([])
            burst_sizes = array([])

        cache["burst_starts"] = burst_starts
        cache["burst_ends"] = burst_ends
        cache["burst_sizes"] = burst_sizes

    sfe = (
        model_params["star_formation_efficiency"]
        * (mstar / 1e9) ** 0.25
        * (1 + exp(mstar / (10 * mgas))) ** -3
        * (1 + redshift) ** -1
    )

    sfr = sfe * mgas

    # is_burst gives an array that's True when t is inside a specific burst
    # if there is a True in there, add the starburst contribution to the sfr
    # note: we don't cache the starting star mass, because we might be
    # encountering a starburst in a mini-step, i.e. the star mass at the start
    # of the burst hasn't been finalized yet. Plus, it's just a few extra
    # polynomial evaluations, not actually that bad for performance
    is_burst = (burst_starts >= t) & (burst_ends <= t)
    if any(is_burst):
        burst_start = burst_starts[is_burst]
        burst_sfr = star_hist(burst_start) * burst_sizes[is_burst]
    else:
        burst_sfr = 0

    return sfr + burst_sfr


def life_from_mass_vec(
    masses, metal_hist, gas_hist, stellar_lifetimes, type_Ia_ratio, t, cache
):
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

    if abs(
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
    ).max() > fp(2e-3):
        soln = root(
            lambda tau: stellar_lifetimes(
                (
                    clip(
                        (metal_hist(t - tau) / gas_hist(t - tau))[:, 0],
                        fp(0.001),
                        fp(0.04),
                    ),
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

    return soln * (1 + type_Ia_ratio)


def supernova_rate(imf, metal_hist, gas_hist, sfr_hist, t, stellar_lifetimes, cache):
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
