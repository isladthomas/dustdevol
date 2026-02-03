from numpy import log10
from dustdevol.generic import fp_zeros
from dustdevol.DeVis2017 import mass_from_life, life_from_mass


def stellar_ejecta(
    model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
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

    # initialize  our integrals
    ejected_gas = fp_zeros(len(mgas))
    ejected_metal = fp_zeros(len(mmetal))
    ejected_dust = fp_zeros(len(mdust))

    # set the maximum mass allowed
    mu = 120

    # store all model params for easier passing to subroutines
    dust_yield_table = model_params["dust_yields"]
    metal_yield_table = model_params["metal_yields"]
    metallicity_cutoffs = model_params["yield_table_z_cutoffs"]
    sn_reduction = model_params["sn_dust_reduction"]
    stellar_lifetimes = model_params["stellar_lifetimes"]

    # determine if high or low metallicity lifetimes are to be used
    if (mmetal[0] / mgas[0]) <= 0.008:
        metallicity = "low"

    else:
        metallicity = "high"

    # set the initial mass for integration to the smallest star that could
    # have died during the sim, or just set it to mu if only stars more massive
    # could have died
    m_min = mass_from_life(times[i], stellar_lifetimes, metallicity)

    if m_min > mu:
        m_min = mu

    # store the current metallicity for future use, or create that variable if
    # needed
    try:
        model_params["z_history"][i, :] = mmetal / mgas[0]
    except KeyError:
        model_params["z_history"] = fp_zeros((len(times), len(mmetal)))

    # finish setting up the integral, done with 500 logarithmically
    # spaced steps across the mass range
    steps = 512
    count = 0
    m = m_min
    dlogm = 0
    logm_new = log10(m) + dlogm
    dm = 10 ** (logm_new) - m
    dlogm = (log10(mu) - log10(m_min)) / steps

    # define some helper functions which do 0-order interpolation for the
    # historical sfr and metallicity
    def z_near(td):
        return model_params["z_history"][(abs(times - td)).argmin()]

    def sfr_near(td):
        return sfr[(abs(times - td)).argmin()]

    # main loop
    while count < steps:
        # increment step num, find the current step size, and find midpoint
        # of this step.
        count += 1
        logm_new = log10(m) + dlogm
        dm = 10.0 ** (logm_new) - m
        mmid = 10.0 ** ((logm_new + log10(m)) / 2.0)

        # calculate lifetime based on midpoint of this step,
        # and use that to find the time when such a star was born
        tau_m = life_from_mass(mmid, stellar_lifetimes, metallicity)
        t_birth = times[i] - tau_m

        # if such a star would need to have been born before the galaxy,
        # ignore it. Otherwise, find out what we get from it exploding
        if t_birth > 0:

            z_birth = z_near(t_birth)
            sfr_birth = sfr_near(t_birth)

            ejected_gas += gas_ejecta(mmid, sfr_birth, imf) * dm
            ejected_metal += (
                metal_ejecta(
                    metal_yield_table,
                    metallicity_cutoffs,
                    mmid,
                    sfr_birth,
                    z_birth,
                    mmetal / mgas[0],
                    imf,
                )
                * dm
            )
            ejected_dust += (
                dust_ejecta(
                    dust_yield_table,
                    metal_yield_table,
                    metallicity_cutoffs,
                    sn_reduction,
                    mmid,
                    sfr_birth,
                    z_birth,
                    mmetal / mgas[0],
                    imf,
                )
                * dm
            )

        # set the left edge of the new step
        m = 10 ** (logm_new)

    return ejected_gas, ejected_metal, ejected_dust


def gas_ejecta(m, sfr, imf):
    """
    calculate gas ejected from the death of stars of mass m,
    i.e. all the mass except what's left in the remnant
    """

    if m >= 120:
        dej = 0

    else:
        dej = (m - remnant_mass(m)) * sfr * imf(m)

    return dej


def metal_ejecta(
    yield_table, metallicity_cutoffs, m, sfr, z_at_birth, metallicity, imf
):
    """
    calculate the metals ejected from the death of stars of mass m,
    composed of all the metals that went into the star when it was first born,
    plus new metals created over its lifespan, including it's death
    """

    if m >= 120:
        dej = 0

    else:
        dej = (
            (
                (m - remnant_mass(m)) * z_at_birth
                + fresh_metals(yield_table, metallicity_cutoffs,
                               m, metallicity)
            )
            * sfr
            * imf(m)
        )

    return dej


def dust_ejecta(
    dust_yield_table,
    metal_yield_table,
    metallicity_cutoffs,
    reduction_factor,
    m,
    sfr,
    z_at_birth,
    metallicity,
    imf,
):
    """
    calculate the dust ejected from the death of stars of mass m,
    composed of a small fraction of the metals that went into the star
    when it was first born (though only for planetary nebula, not supernovae),
    as well as new dust formed during it's death. Additionally, assume that,
    for stars greater than 40 MSol, no dust is released. This is because
    these stars do not go supernova or form planetary nebula, but rather
    collapse into a black hole: the only thing they release is gas and
    metals in strong stellar winds just before the collapse.

    NOTE: dust doesn't survive inside of a star, so we don't need to worry
    about that, however dust can still be created in stellar winds while the
    star is still alive. However, this only happens to a substantial degree in
    exceedingly rare stars, and even then each of these stars doesn't produce
    that much dust during it's life.
    """

    # determine if the star is a planetary nebula (recycles) or SN (demonic)
    if m <= 8.0:
        delta_LIMS = 0.15

    else:
        delta_LIMS = 0

    # further determine if the star becomes a black hole, and produces no dust
    if m >= 40:
        dej = 0

    else:
        dej_fresh = (
            fresh_dust(
                dust_yield_table,
                metal_yield_table,
                metallicity_cutoffs,
                reduction_factor,
                m,
                metallicity,
            )
            * sfr
            * imf(m)
        )
        dej_recycled = (m - remnant_mass(m)) * \
            z_at_birth[0] * delta_LIMS * sfr * imf(m)

        dej = dej_fresh + dej_recycled

    return dej


def remnant_mass(m):
    """
    calculates how much mass of the star remains in stellar remnants.
    """

    if m <= 9.0:
        rem_mass = 0.106 * m + 0.446

    elif m < 25.0:
        rem_mass = 1.5

    else:
        rem_mass = 0.61 * m - 13.75

    return rem_mass


def fresh_metals(yield_table, metallicity_cutoffs, m, metallicity):
    """
    nab the amount of metals generated in the death of a star of mass m
    from the yield table
    """

    yields = yield_table[abs(yield_table[:, 0] - m).argmin(), 1:]
    stepsize = len(metallicity)

    for i, cutoff in enumerate(metallicity_cutoffs):
        if metallicity[0] <= cutoff:
            return yields[i * stepsize: (i + 1) * stepsize]


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
    if m <= 8:
        dust_mass = (
            0.15
            * fresh_metals(metal_yield_table, metallicity_cutoffs, m, metallicity)[0]
        )

    elif m <= 40:
        dust_mass = (
            dust_yield_table[abs(dust_yield_table[:, 0] - m).argmin(), 1]
            / reduction_factor
        )
    else:
        dust_mass = 0

    return dust_mass
