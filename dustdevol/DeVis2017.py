from dustdevol.generic import fp, fp_zeros
from numpy import where, log10, logspace, linspace, vectorize, diff


def xSFR_inflow(model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust):
    """
    calculate gas inflow, and try to calculate metal and dust,
    if inflow metal and dust frac are not specified, assume 0
    requires:
        - \"inflow_xSFR\": multiple of SFR to calculate inflows
                           should have same shape as init_gas
    optional:
        - \"inflow_metal\": fraction of inflows in the form of metal
                            should have same shape as init_metal
        - \"inflow_dust\": fraction of inflows in the form of dust
                           should have same shape as init_dust
    """

    gas_inflow = sfr[i] * model_params["inflow_xSFR"]

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
    model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
):
    """
    calculate gas inflow, and try to calculate metal and dust,
    if inflow metal and dust frac are not specified, assume 0
    requires:
        - \"inflow_xSFR\": multiple of SFR to calculate inflows
                           should have same shape as init_gas
    optional:
        - \"inflow_metal\": fraction of inflows in the form of metal
                            should have same shape as init_metal
        - \"inflow_dust\": fraction of inflows in the form of dust
                           should have same shape as init_dust
    """

    gas_outflow = sfr[i] * model_params["outflow_xSFR"]

    # if metal and dust frac not specified, set to zero
    # (so the error stops happening and we can go on quicker)
    try:
        metal_outflow = gas_outflow * model_params["outflow_metal"]
    except KeyError:
        model_params["outflow_metal"] = fp_zeros(len(mmetal))
        metal_outflow = fp_zeros(len(mmetal))

    try:
        dust_outflow = gas_outflow * model_params["outflow_dust"]
    except KeyError:
        model_params["outflow_dust"] = fp_zeros(len(mdust))
        dust_outflow = fp_zeros(len(mdust))

    return gas_outflow, metal_outflow, dust_outflow


def grain_growth(
    model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
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
        model_params["grain_growth_epsilon"], mgas, sfr[i], mmetal[0], mdust
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
    model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
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

    sn_rate = supernova_rate(
        sfr, imf, times, i, model_params["stellar_lifetimes"])

    t_des = destruction_timescale(
        model_params["sn_destruction"], mgas, sn_rate)

    mdust_des = mdust * (1 - model_params["cold_fraction"]) * t_des**-1

    return mdust_des


def destruction_timescale(destruction, mgas, sn_rate):
    """
    calculate timescale, in Gyrs, for 1 solar mass of gas to be swept up in
    a supernova
    """

    t_destroy = mgas / (destruction * sn_rate)

    return t_destroy


def supernova_rate(sfr, imf, times, i, stellar_lifetimes):
    """
    calculate rate of supernova events in SN / Gyr, assuming stars
    that go supernova have a short enough lifespan to be born and die
    in a single timestep (30-50 Myr)
    """

    # set up integral over mass
    sn_rate = 0
    dm = 0.01

    # find the least massive star that could have been born and died
    # during the sim so far, clamped at 8 Msol (cutoff for SN)
    m = mass_from_life(times[i], stellar_lifetimes, "high")
    m = max(8, m)

    # integrate over imf between 8 and 40 msol, to find number of supernovae
    # per solar mass of stars formed. At 10 Msol, increase step size
    while m < 10.0:
        sn_rate += imf(m) * dm
        m += dm

    dm = 0.5

    while m < 40.0:
        sn_rate += imf(m) * dm
        m += dm

    # multiply by sfr to get final rate
    sn_rate = sfr[i] * sn_rate

    return sn_rate


def mass_from_life(t, stellar_lifetimes, metallicity):
    """
    helper function which uses 0-order interpolation to find the
    mass in Msol of a star with (at most) a given lifetime in Gyr
    """

    # find diff between requested lifetime and lifetime of each mass
    if metallicity == "high":
        diffs = t - stellar_lifetimes[:, 2]

    else:
        diffs = t - stellar_lifetimes[:, 1]

    # if the difference is negative, then the lifetime of such a star is
    # longer than the requested lifetime, so send those off, and *then* find
    # the closest value
    arg = where(diffs > 0, diffs, fp("inf")).argmin()

    return stellar_lifetimes[arg, 0]


def life_from_mass(m, stellar_lifetimes, metallicity):
    """
    helper function which uses 0-order interpolation to find the lifetime
    in Gyr given a mass in Msol
    """

    arg = (abs(stellar_lifetimes[:, 0] - m)).argmin()

    if metallicity == "high":
        return stellar_lifetimes[arg, 2]

    else:
        return stellar_lifetimes[arg, 1]


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
    steps = 65536
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
        dm = 10 ** (logm_new) - m
        mmid = 10 ** ((logm_new + log10(m)) / 2)

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


def fast_ejecta_lin(model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust):
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

        model_params["ejecta_masses"] = logspace(0.8, 120, 65537)

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
        remnants = vectorize(remnant_mass)(masses)
        model_params["ejecta_vals"] = masses - remnants
        ejecta = model_params["ejecta_vals"]

    # determine if high or low metallicity lifetimes are to be used
    if (mmetal[0] / mgas[0]) <= 0.008:
        metallicity = "low"

    else:
        metallicity = "high"

    lifetimes = vectorize(life_from_mass, excluded={1, 2})(
        masses, stellar_lifetimes, metallicity
    )

    d_masses = where(times[i] > lifetimes, d_masses, 0)

    # store the current metallicity for future use, or create that variable if
    # needed
    try:
        model_params["z_history"][i, :] = mmetal / mgas[0]
    except KeyError:
        model_params["z_history"] = fp_zeros((len(times), len(mmetal)))
        model_params["z_history"][i, :] = mmetal / mgas[0]

    # define some helper functions which do 0-order interpolation for the
    # historical sfr and metallicity
    def z_near(td):
        return model_params["z_history"][(abs(times - td)).argmin()]

    def sfr_near(td):
        return sfr[(abs(times - td)).argmin()]

    # create arrays for historical metallicity and sfr
    z_at_birth = fp_zeros((len(lifetimes), len(mmetal)))
    sfr_vals = fp_zeros(len(lifetimes))
    for i, t in enumerate(times[i] - lifetimes):
        z_at_birth[i, :] = z_near(t)
        sfr_vals[i] = sfr_near(t)

    # calculate all our ejecta
    ejected_gas = (ejecta * sfr_vals * imf_vals * d_masses).sum()

    fresh_metal_ejecta = fp_zeros((len(lifetimes), len(mmetal)))
    for i, mass in enumerate(masses):
        fresh_metal_ejecta[i, :] = fresh_metals(
            metal_yield_table, metallicity_cutoffs, mass, mmetal / mgas[0]
        )
    old_metal_ejecta = ejecta[:, None] * z_at_birth
    ejected_metal = (
        (fresh_metal_ejecta + old_metal_ejecta)
        * sfr_vals[:, None]
        * imf_vals[:, None]
        * d_masses[:, None]
    ).sum()

    fresh_dust_ejecta = vectorize(fresh_dust, excluded={0, 1, 2, 3, 5})(
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
    ).sum()

    return ejected_gas, ejected_metal, ejected_dust


def fast_ejecta_log(model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust):
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

        model_params["ejecta_masses"] = linspace(0.8, 120, 65537)

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
        remnants = vectorize(remnant_mass)(masses)
        model_params["ejecta_vals"] = masses - remnants
        ejecta = model_params["ejecta_vals"]

    # determine if high or low metallicity lifetimes are to be used
    if (mmetal[0] / mgas[0]) <= 0.008:
        metallicity = "low"

    else:
        metallicity = "high"

    lifetimes = vectorize(life_from_mass, excluded={1, 2})(
        masses, stellar_lifetimes, metallicity
    )

    d_masses = where(times[i] > lifetimes, d_masses, 0)

    # store the current metallicity for future use, or create that variable if
    # needed
    try:
        model_params["z_history"][i, :] = mmetal / mgas[0]
    except KeyError:
        model_params["z_history"] = fp_zeros((len(times), len(mmetal)))
        model_params["z_history"][i, :] = mmetal / mgas[0]

    # define some helper functions which do 0-order interpolation for the
    # historical sfr and metallicity
    def z_near(td):
        return model_params["z_history"][(abs(times - td)).argmin()]

    def sfr_near(td):
        return sfr[(abs(times - td)).argmin()]

    # create arrays for historical metallicity and sfr
    z_at_birth = fp_zeros((len(lifetimes), len(mmetal)))
    sfr_vals = fp_zeros(len(lifetimes))
    for i, t in enumerate(times[i] - lifetimes):
        z_at_birth[i, :] = z_near(t)
        sfr_vals[i] = sfr_near(t)

    # calculate all our ejecta
    ejected_gas = (ejecta * sfr_vals * imf_vals * d_masses).sum()

    fresh_metal_ejecta = fp_zeros((len(lifetimes), len(mmetal)))
    for i, mass in enumerate(masses):
        fresh_metal_ejecta[i, :] = fresh_metals(
            metal_yield_table, metallicity_cutoffs, mass, mmetal / mgas[0]
        )
    old_metal_ejecta = ejecta[:, None] * z_at_birth
    ejected_metal = (
        (fresh_metal_ejecta + old_metal_ejecta)
        * sfr_vals[:, None]
        * imf_vals[:, None]
        * d_masses[:, None]
    ).sum()

    fresh_dust_ejecta = vectorize(fresh_dust, excluded={0, 1, 2, 3, 5})(
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
    ).sum()

    return ejected_gas, ejected_metal, ejected_dust
