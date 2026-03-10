from dustdevol.generic import fp
from numpy import (
    logspace,
    where,
    diff,
    log10,
    searchsorted,
    clip,
)
from scipy.optimize.elementwise import find_root


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
        tau0 = cache["ejecta_lifetimes"]
    except KeyError:
        cache["ejecta_lifetimes"] = stellar_lifetimes((fp(0), masses))
        tau0 = cache["ejecta_lifetimes"]

    retry = abs(
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
    ) > fp(2e-3)
    if any(retry):
        tau0[retry] = find_root(
            lambda tau, mass: stellar_lifetimes(
                (
                    clip(
                        (metal_hist(t - tau) / gas_hist(t - tau))[:, 0],
                        fp(0.001),
                        fp(0.04),
                    ),
                    mass,
                )
            )
            - tau,
            [tau0[retry] / 1.5, tau0[retry] * 1.5],
            args=(masses[retry],),
            tolerances={"xatol": fp(2e-3)},
        ).x

    cache["ejecta_lifetimes"] = tau0

    return tau0


def remnant_mass(m):
    """
    Calculates how much mass of the star remains in stellar remnants using
    the prescription of Ferreras and Silk 2000

    Parameters
    ----------
    m : ndarray
        masses for which the remnant mass is to be calculated

    Returns
    -------
    out: ndarray
         remnant masses in Msol for the stars
    """

    rem_mass = where(m < fp(25), fp(1.5), fp(0.61) * m - fp(13.75))
    rem_mass[m <= fp(9)] = (fp(0.106) * m + fp(0.446))[m <= fp(9)]

    return rem_mass


def fresh_metals(yield_table, metallicity_cutoffs, masses, metallicity):
    """
    Nab the amount of metals generated in the death of a star of mass m
    from the yield table

    Parameters
    ----------
    yield_table : 2D array
                  array with the first column containing masses, with
                  subsequent columns being the metal yields for progressiely
                  higher metallicities.
    metallicity_cutoffs : 1D array
                          list of cutoff values for the metal yield table
    masses : array_like
             masses for which the fresh metal yields are to be evaluated
    metallicity : float
                  current metallicity

    Returns
    -------
    out : ndarray
          array with the shape of `masses` but with an additional dimension,
          giving the metal yield from stars of the corresponding mass.
    """

    i = searchsorted(metallicity_cutoffs, metallicity)[0]
    stepsize = len(metallicity)

    masses_half = yield_table[:-1, 0] / fp(2) + yield_table[1:, 0] / fp(2)
    eff_indices = searchsorted(masses_half, masses)
    eff_indices = clip(eff_indices, 0, len(yield_table[:, 0]) - 1)
    return yield_table[eff_indices, i * stepsize + 1 : (i + 1) * stepsize + 1]


def fresh_dust(
    eff_table,
    ejected_metals,
    reduction_factor,
    masses,
):
    """
    Nab the amount of dust generated in the death of a star of mass m,
    either from a yield table given in % of metals, if the star goes SN,
    or assuming a constant fraction of 15% if not. Assumes zero dust from
    black hole progenitors, assumed to be all stars with initial mass above
    40 Msol.

    Parameters
    ----------
    eff_table : 2D array
                2D array where first column gives list of masses and second
                gives dust formation efficieny
    ejected_metals : array_like, shape (m,)
                     array of metals ejected from the death of star of mass
                     corresponding to those in `masses`
    reduction_factor : float
                       constant factor to divide SN dust by
    masses : array_like, shape(m,)
             array of progenitor masses to find the dust output of

    Results
    -------
    out : ndarray, shape (m,)
          array of dust output from stars of each mass.
    """

    masses_half = eff_table[:-1, 0] / fp(2) + eff_table[1:, 0] / fp(2)
    eff_indices = searchsorted(masses_half, masses, side="left")
    eff_indices = clip(eff_indices, 0, len(eff_table[:, 0]) - 1)
    dust_eff = eff_table[eff_indices, 1] / reduction_factor
    dust_eff[masses <= fp(8)] = fp(0.15)
    dust_eff[masses > fp(40)] = fp(0)

    return dust_eff * ejected_metals


def stellar_ejecta(
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
    Calculate the gas, metals, and dust emmitted from dying stars, given
    a function for mass of stellar remnants as well as output tables for
    metal and dust yields. In essence, convolves the past SFR with the IMF
    and a yield function to find how much gas/metal/dust is beind shot out now

    Parameters
    ----------
    model_params : dict
        - \"dust_yields\": table where each row gives a mass in Msol, and the
                           dust *created*, not recycled, when such a star dies
        - \"metal_yields\": table where each row gives a mass in Msol, followed
                            by several entries giving the metals created when
                            such a star dies, ordered the same as init_metals,
                            repeated for each metallicity level
        - \"yield_table_z_cutoffs\": list giving the cutoff for each
                                     metallicity level in the metal_yields table
        - \"sn_dust_reduction\": factor which divides dust created in supernovae
        - \"stellar_lifetimes\": table where each row gives, in order
                                 the mass of a star in Msol, the lifetime
                                 of such a star in Gyrs in a low metallicity
                                 (Z < 0.008) environment, and the lifetime in
                                 a high metallicity (Z >= 0.008) environment

    Results
    -------
    out : (g,), (m,), (d,)
          gas, metals, and dust ejected from dying stars in Msol/Gyr
    """

    # store all model params for easier passing to subroutines
    dust_yield_table = model_params["dust_yields"]
    metal_yield_table = model_params["metal_yields"]
    metallicity_cutoffs = model_params["yield_table_z_cutoffs"]
    sn_Ia_dust_yields = model_params["type_Ia_dust_yields"]
    sn_Ia_metal_yields = model_params["type_Ia_metal_yields"]
    sn_Ia_cutoffs = model_params["type_Ia_yield_table_z_cutoffs"]
    sn_reduction = model_params["sn_dust_reduction"]
    stellar_lifetimes = model_params["stellar_lifetimes"]
    sn_Ia_lifetimes = model_params["type_Ia_delays"]
    sn_Ia_prob = model_params["type_Ia_probability"]

    # grab everything precomputable, and precompute it if not
    # specifically, the masses we sample, and the imfs, ejecta (m - rem)
    # and the size of the window at each mass
    try:
        masses = cache["ejecta_masses"]
        ejecta = cache["ejecta_vals"]
        imf_vals = cache["imf_values"]
        d_masses = cache["d_masses"]
        stars_per_gen = cache["stars_per_gen"]

    except KeyError:

        cache["ejecta_masses"] = logspace(log10(0.8), log10(120), 513, dtype=fp)

        # get mass windows
        masses = cache["ejecta_masses"]
        cache["d_masses"] = diff(masses)
        d_masses = cache["d_masses"]

        # switch "masses" to the midpoints, instead of left edges
        cache["ejecta_masses"] = masses[:-1] + (d_masses / fp(2))
        masses = cache["ejecta_masses"]

        # calcualte imf and ejecta at midpoints
        cache["imf_values"] = imf(masses) * where(
            masses <= 8, 1 - sn_Ia_prob, 1
        )  # don't double_count Ia's
        imf_vals = cache["imf_values"]
        remnants = remnant_mass(masses)
        cache["ejecta_vals"] = masses - remnants
        ejecta = cache["ejecta_vals"]

        # calculate total number of stars per solar mass of formed stars
        # important for calculating type Ia rates
        tot_masses = logspace(log10(0.1), log10(120), 513, dtype=fp)
        tot_d_masses = diff(tot_masses)
        tot_masses = tot_masses[:-1] + (tot_d_masses / fp(2))
        cache["stars_per_gen"] = (imf(tot_masses) * tot_d_masses).sum()
        stars_per_gen = cache["stars_per_gen"]

    # determine if high or low metallicity lifetimes are to be used
    lifetimes = life_from_mass_vec(
        masses, metal_hist, gas_hist, stellar_lifetimes, t, cache
    )

    d_masses = where(t > lifetimes, d_masses, fp(0))

    # create arrays for historical metallicity and sfr
    z_at_birth = metal_hist(t - lifetimes) / gas_hist(t - lifetimes)
    sfr_vals = sfr_hist(t - lifetimes)

    # calculate all our ejecta
    ejected_gas = (ejecta * sfr_vals * imf_vals * d_masses).sum(axis=0)

    fresh_metal_ejecta = fresh_metals(
        metal_yield_table, metallicity_cutoffs, masses, mmetal / mgas[0]
    )
    old_metal_ejecta = ejecta[:, None] * z_at_birth
    ejected_metal = (
        (fresh_metal_ejecta + old_metal_ejecta)
        * sfr_vals[:, None]
        * imf_vals[:, None]
        * d_masses[:, None]
    ).sum(axis=0)

    fresh_dust_ejecta = fresh_dust(
        dust_yield_table,
        (fresh_metal_ejecta + old_metal_ejecta)[:, 0],
        sn_reduction,
        masses,
    )
    ejected_dust = (fresh_dust_ejecta * sfr_vals * imf_vals * d_masses).sum(axis=0)
    # calculate type Ia contribution
    sn_Ia_rate = (
        stars_per_gen
        * sn_Ia_prob
        * sfr_vals[masses <= 8]
        * sn_Ia_lifetimes(lifetimes[masses <= 8])
        * where(
            t > lifetimes[masses <= 8], -diff(lifetimes)[masses[:-1] <= 8], fp(0)
        )
    )
    cache["Ia_rate"] = sn_Ia_rate.sum()
    ejected_gas += (ejecta[masses <= 8] * sn_Ia_rate).sum(axis=0)

    sn_Ia_fresh_metal = fresh_metals(
        sn_Ia_metal_yields, sn_Ia_cutoffs, masses[masses <= 8], mmetal / mgas[0]
    )
    ejected_metal += (
        (sn_Ia_fresh_metal + old_metal_ejecta[masses <= 8]) * sn_Ia_rate[:, None]
    ).sum(axis=0)

    sn_Ia_fresh_dust = fresh_dust(
        sn_Ia_dust_yields,
        (sn_Ia_fresh_metal + old_metal_ejecta[masses <= 8])[:, 0],
        sn_reduction,
        masses[masses <= 8],
    )
    ejected_dust += (sn_Ia_fresh_dust * sn_Ia_rate).sum(axis=0)

    return ejected_gas, ejected_metal, ejected_dust
