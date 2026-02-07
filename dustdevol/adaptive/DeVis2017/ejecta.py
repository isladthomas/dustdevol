from numpy import array, logspace, where, diff, vectorize, log10, searchsorted, clip
from dustdevol.adaptive.generic import fp_zeros


def life_from_mass_vec(m, stellar_lifetimes, metallicity):
    """
    helper function which uses 0-order interpolation to find the lifetime
    in Gyr given a mass in Msol
    """

    lifetimes = fp_zeros(len(m))

    for i, mass in enumerate(m):
        arg = (abs(stellar_lifetimes[:, 0] - mass)).argmin()

        if metallicity == "high":
            lifetimes[i] = stellar_lifetimes[arg, 2]

        else:
            lifetimes[i] = stellar_lifetimes[arg, 1]

    return lifetimes


def remnant_mass(m):
    """
    calculates how much mass of the star remains in stellar remnants.
    """

    rem_mass = where(m < 25, 1.5, 0.61 * m - 13.75)
    rem_mass[m <= 9] = (0.106 * m + 0.446)[m <= 9]

    return rem_mass


def fresh_metals(yield_table, metallicity_cutoffs, masses, metallicity):
    """
    nab the amount of metals generated in the death of a star of mass m
    from the yield table
    """

    i = searchsorted(metallicity_cutoffs, metallicity)[0]
    stepsize = len(metallicity)

    masses_half = yield_table[:-1, 0] / 2 + yield_table[1:, 0] / 2
    eff_indices = searchsorted(masses_half, masses)
    eff_indices = clip(eff_indices, 0, len(yield_table[:, 0]) - 1)
    return yield_table[eff_indices, i * stepsize + 1: (i + 1) * stepsize + 1]


def fresh_dust(
    eff_table,
    ejected_metals,
    reduction_factor,
    masses,
):
    """
    nab the amount of dust generated in the death of a star of mass m,
    either from the yield table, if the star goes supernova,
    or from a fraction of the metals generated, if the star becomes a PN
    """

    masses_half = eff_table[:-1, 0] / 2 + eff_table[1:, 0] / 2
    eff_indices = searchsorted(masses_half, masses, side="left")
    eff_indices = clip(eff_indices, 0, len(eff_table[:, 0]) - 1)
    dust_eff = eff_table[eff_indices, 1] / reduction_factor
    dust_eff[masses <= 8] = 0.15
    dust_eff[masses > 40] = 0

    return dust_eff * ejected_metals


def fast_ejecta(
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
    calculate the gas, metals, and dust emmitted from dying stars, given
    a function for mass of stellar remnants as well as output tables for
    metal and dust yields. In essence, convolves the past SFR with the IMF
    and a yield function to find how much gas/metal/dust is beind shot out now
    requires:
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
    NOTE: does modify model_params, storing an additional value with key
    \"z_history\", which allows the function to access historical metallicities
    """

    # store all model params for easier passing to subroutines
    dust_yield_table = model_params["dust_yields"]
    metal_yield_table = model_params["metal_yields"]
    metallicity_cutoffs = model_params["yield_table_z_cutoffs"]
    sn_reduction = model_params["sn_dust_reduction"]
    stellar_lifetimes = model_params["stellar_lifetimes"]

    # grab everything precomputable, and precompute it if not
    # specifically, the masses we sample, and the imfs, ejecta (m - rem)
    # and the size of the window at each mass
    try:
        masses = model_params["ejecta_masses"]
        ejecta = model_params["ejecta_vals"]
        imf_vals = model_params["imf_values"]
        d_masses = model_params["d_masses"]

    except KeyError:

        model_params["ejecta_masses"] = logspace(log10(0.8), log10(120), 513)

        # get mass windows
        masses = model_params["ejecta_masses"]
        model_params["d_masses"] = diff(masses)
        d_masses = model_params["d_masses"]

        # switch "masses" to the midpoints, instead of left edges
        model_params["ejecta_masses"] = masses[:-1] + (d_masses / 2)
        masses = model_params["ejecta_masses"]

        # calcualte imf and ejecta at midpoints
        model_params["imf_values"] = vectorize(imf)(masses)
        imf_vals = model_params["imf_values"]
        remnants = remnant_mass(masses)
        model_params["ejecta_vals"] = masses - remnants
        ejecta = model_params["ejecta_vals"]

    # determine if high or low metallicity lifetimes are to be used
    if (mmetal[0] / mgas[0]) <= 0.008:
        metallicity = "low"

    else:
        metallicity = "high"

    lifetimes = life_from_mass_vec(masses, stellar_lifetimes, metallicity)

    d_masses = where(t > lifetimes, d_masses, 0)

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
    ejected_dust = (fresh_dust_ejecta * sfr_vals *
                    imf_vals * d_masses).sum(axis=0)

    return ejected_gas, ejected_metal, ejected_dust


"""
def Gauss_Kronrod_ejecta(
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
"""
    calculate the gas, metals, and dust emmitted from dying stars, given
    a function for mass of stellar remnants as well as output tables for
    metal and dust yields. In essence, convolves the past SFR with the IMF
    and a yield function to find how much gas/metal/dust is beind shot out now
    requires:
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
    Uses Gauss-Kronrod integration to (ideally) lower the workload and also
    give an error estimate. If the error is too high, subdivides interval.
    """
"""

    # store all model params for easier passing to subroutines
    dust_yield_table = model_params["dust_yields"]
    metal_yield_table = model_params["metal_yields"]
    metallicity_cutoffs = model_params["yield_table_z_cutoffs"]
    sn_reduction = model_params["sn_dust_reduction"]
    stellar_lifetimes = model_params["stellar_lifetimes"]

    # grab everything precomputable, and precompute it if not
    # masses we sample, imf at each mass, gas ejecta at each mass,
    # as well as weights
    try:
        masses = cache["ejecta_masses"]
        ejecta = cache["ejecta_vals"]
        imf_vals = cache["imf_values"]
        gauss_weights = cache["gauss_weights"]
        kronrod_weights = cache["kronrod_weights"]

    except KeyError:

        cache["ejecta_masses"] = ((sample_points_pre + 1) / 2) * (119.2) + 0.8
        masses = cache["ejecta_masses"]
        remnants = remnant_mass(masses)
        cache["ejecta_vals"] = masses - remnants
        cache["imf_values"] = array([imf(xi) for xi in masses])
        cache["gauss_weights"] = gauss_weights_pre * 59.6
        cache["kronrod_weights"] = kronrod_weights_pre * 59.6

        ejecta = cache["ejecta_vals"]
        imf_vals = cache["imf_values"]
        gauss_weights = cache["gauss_weights"]
        kronrod_weights = cache["kronrod_weights"]

    # determine if high or low metallicity lifetimes are to be used
    if (mmetal[0] / mgas[0]) <= 0.008:
        metallicity = "low"

    else:
        metallicity = "high"

    lifetimes = life_from_mass_vec(masses, stellar_lifetimes, metallicity)

    d_masses = where(t > lifetimes, d_masses, 0)

    # create arrays for historical metallicity and sfr
    z_at_birth = fp_zeros((len(lifetimes), len(mmetal)))
    sfr_vals = fp_zeros(len(lifetimes))
    for j, t in enumerate(t - lifetimes):
        if t >= 0:
            z_at_birth[j, :] = metal_hist(t) / gas_hist(t)
            sfr_vals[j] = sfr_hist(t)

    # calculate all our ejecta
    ejected_gas = (ejecta * sfr_vals * imf_vals * d_masses).sum(axis=0)

    fresh_metal_ejecta = fp_zeros((len(lifetimes), len(mmetal)))
    for j, mass in enumerate(masses):
        if d_masses[j] != 0:
            fresh_metal_ejecta[j, :] = fresh_metals(
                metal_yield_table, metallicity_cutoffs, mass, mmetal / mgas[0]
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
        metal_yield_table,
        metallicity_cutoffs,
        sn_reduction,
        masses,
        mmetal / mgas[0],
    )
    old_dust_ejecta = ejecta * z_at_birth[:, 0] * where(masses <= 8, 0.15, 0)
    ejected_dust = (
        (fresh_dust_ejecta + old_dust_ejecta)
        * sfr_vals
        * imf_vals
        * d_masses
        * where(masses < 40, 1, 0)
    ).sum(axis=0)

    return ejected_gas, ejected_metal, ejected_dust


# Sample points and weights for 15-point Gauss-Kronrod integration.
# Based on integrating on [-1,1], needs to be rescaled to new bounds
sample_points_pre = [
    0.991455371120813,
    0.949107912342759,
    0.864864423359769,
    0.741531185599394,
    0.586087235467691,
    0.405845151377397,
    0.207784955007898,
    0.000000000000000,
    -0.207784955007898,
    -0.405845151377397,
    -0.586087235467691,
    -0.741531185599394,
    -0.864864423359769,
    -0.949107912342759,
    -0.991455371120813,
]

gauss_weights_pre = [
    0,
    0.129484966168870,
    0,
    0.279705391489277,
    0,
    0.381830050505119,
    0,
    0.417959183673469,
    0,
    0.381830050505119,
    0,
    0.279705391489277,
    0,
    0.129484966168870,
    0,
]

kronrod_weights_pre = [
    0.022935322010529,
    0.063092092629979,
    0.104790010322250,
    0.129484966168870,
    0.169004726639267,
    0.190350578064785,
    0.204432940075298,
    0.209482141084728,
    0.204432940075298,
    0.190350578064785,
    0.169004726639267,
    0.129484966168870,
    0.104790010322250,
    0.063092092629979,
    0.022935322010529,
]
"""
