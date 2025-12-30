from dustdevol.generic import fp, fp_zeros
from numpy import where, log10


def xSFR_inflow(model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust):
    """calc gas inflow, try with metal and dust,
    if inflow metal and dust frac are not specified, assume 0"""

    gas_inflow = sfr[i] * model_params["inflow_xSFR"]
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
    """calc gas outflow, try with metal and dust,
    if inflow metal and dust frac are not specified, assume 0"""

    gas_outflow = sfr[i] * model_params["outflow_xSFR"]
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

    time_gg = grow_timescale(
        model_params["grain_growth_epsilon"], mgas, sfr[i], mmetal[0], mdust
    )  # convert grain growth timescale to Gyrs
    mdust_gg = (
        mdust * model_params["cold_fraction"] *
        (1.0 - (mdust / mmetal[0])) * time_gg**-1
    )
    if any(mdust_gg != mdust_gg):
        mdust_gg = fp_zeros(len(mdust))
    return mdust_gg


def grow_timescale(e, mgas, sfr, mmetal, mdust):

    sfr_yrs = sfr
    t_grow = (mgas**2) / (e * mmetal * sfr_yrs)

    return t_grow


def dust_destruction(
    model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
):
    sn_rate = supernova_rate(
        sfr, imf, times, i, model_params["stellar_lifetimes"])
    t_des = destruction_timescale(
        model_params["sn_destruction"], mgas, sn_rate)
    mdust_des = mdust * (1 - model_params["cold_fraction"]) * t_des**-1

    return mdust_des


def destruction_timescale(destruction, mgas, sn_rate):
    sn_rate = sn_rate
    t_destroy = mgas / (destruction * sn_rate)
    return t_destroy


def supernova_rate(sfr, imf, times, i, stellar_lifetimes):
    sn_rate = 0
    dm = 0.01
    m = mass_from_life(times[i], stellar_lifetimes, "high")
    m = max(8, m)
    while m < 10.0:
        sn_rate += imf(m) * dm
        m += dm
    dm = 0.5
    while m < 40.0:
        sn_rate += imf(m) * dm
        m += dm
    sn_rate = sfr[i] * sn_rate
    return sn_rate


def mass_from_life(t, stellar_lifetimes, metallicity):
    if metallicity == "high":
        diffs = stellar_lifetimes[:, 2] - t
    else:
        diffs = stellar_lifetimes[:, 1] - t
    arg = where(diffs > 0, diffs, fp("inf")).argmin()
    return stellar_lifetimes[arg, 0]


def life_from_mass(m, stellar_lifetimes, metallicity):
    arg = (abs(stellar_lifetimes[:, 0] - m)).argmin()
    if metallicity == "high":
        return stellar_lifetimes[arg, 2]
    else:
        return stellar_lifetimes[arg, 1]


def stellar_ejecta(
    model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
):
    ejected_gas = fp_zeros(len(mgas))
    ejected_metal = fp_zeros(len(mmetal))
    ejected_dust = fp_zeros(len(mdust))

    mu = 120

    dust_yield_table = model_params["dust_yields"]
    metal_yield_table = model_params["metal_yields"]
    metallicity_cutoffs = model_params["yield_table_z_cutoffs"]
    sn_reduction = model_params["sn_dust_reduction"]
    stellar_lifetimes = model_params["stellar_lifetimes"]

    if (mmetal[0] / mgas[0]) <= 0.008:
        metallicity = "low"

    else:
        metallicity = "high"
    m_min = mass_from_life(times[i], stellar_lifetimes, metallicity)

    if m_min > mu:
        m_min = mu

    try:
        model_params["z_history"][i, :] = mmetal / mgas[0]
    except KeyError:
        model_params["z_history"] = fp_zeros((len(times), len(mmetal)))

    steps = 500
    count = 0
    m = m_min
    dlogm = 0
    logm_new = log10(m) + dlogm
    dm = 10 ** (logm_new) - m
    dlogm = (log10(mu) - log10(m_min)) / steps

    def z_near(td):
        return model_params["z_history"][(abs(times - td)).argmin()]

    def sfr_near(td):
        return sfr[(abs(times - td)).argmin()]

    while count < steps:
        count += 1
        logm_new = log10(m) + dlogm
        dm = 10 ** (logm_new) - m
        mmid = 10 ** ((logm_new + log10(m)) / 2)

        tau_m = life_from_mass(mmid, stellar_lifetimes, metallicity)
        t_birth = times[i] - tau_m

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

        m = 10 ** (logm_new)

    return ejected_gas, ejected_metal, ejected_dust


def gas_ejecta(m, sfr, imf):
    if m >= 120:
        dej = 0
    else:
        dej = (m - remnant_mass(m)) * sfr * imf(m)
    return dej


def metal_ejecta(
    yield_table, metallicity_cutoffs, m, sfr, z_at_birth, metallicity, imf
):
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
    if m <= 8.0:
        delta_LIMS = 0.15
    else:
        delta_LIMS = 0
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
        dej_recycled = (m - remnant_mass(m)) * z_at_birth[0] * \
            delta_LIMS * sfr * imf(m)
        dej = dej_fresh + dej_recycled
    return dej


def remnant_mass(m):
    if m <= 9.0:
        rem_mass = 0.106 * m + 0.446
    elif m < 25.0:
        rem_mass = 1.5
    else:
        rem_mass = 0.61 * m - 13.75
    return rem_mass


def fresh_metals(yield_table, metallicity_cutoffs, m, metallicity):
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
    if m <= 8:
        dust_mass = 0.15 * fresh_metals(
            metal_yield_table, metallicity_cutoffs, m, metallicity
        )[0]
    elif m <= 40:
        dust_mass = (
            dust_yield_table[abs(dust_yield_table[:, 0] - m).argmin(), 1]
            / reduction_factor
        )
    return dust_mass
