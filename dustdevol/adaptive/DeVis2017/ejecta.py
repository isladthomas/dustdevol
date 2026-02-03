from numpy import linspace, logspace, where, diff, vectorize, log10
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


def fresh_metals(yield_table, metallicity_cutoffs, m, metallicity):
    """
    nab the amount of metals generated in the death of a star of mass m
    from the yield table
    """

    yields = yield_table[abs(yield_table[:, 0] - m).argmin(), 1:]
    stepsize = len(metallicity)

    for j, cutoff in enumerate(metallicity_cutoffs):
        if metallicity[0] <= cutoff:
            return yields[j * stepsize: (j + 1) * stepsize]


def fresh_dust(
    dust_yield_table,
    metal_yield_table,
    metallicity_cutoffs,
    reduction_factor,
    m,
    metallicity,
):
    """
    nab the amount of dust generated in the death of a star of mass m,
    either from the yield table, if the star goes supernova,
    or from a fraction of the metals generated, if the star becomes a PN
    """
    dust_mass = fp_zeros(len(m))

    for i, mass in enumerate(m):

        if mass <= 8:
            dust_mass[i] = (
                0.15
                * fresh_metals(metal_yield_table, metallicity_cutoffs, mass, metallicity)[
                    0
                ]
            )

        elif mass <= 40:
            dust_mass[i] = (
                dust_yield_table[abs(dust_yield_table[:, 0] - mass).argmin(), 1]
                / reduction_factor
            )

    return dust_mass


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
