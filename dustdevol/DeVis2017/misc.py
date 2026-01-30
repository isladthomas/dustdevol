from numpy import where
from dustdevol.generic import fp, fp_zeros

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
        metal_outflow = (mmetal / mgas[0]) * gas_outflow * model_params["outflow_metal"]
    except KeyError:
        model_params["outflow_metal"] = fp_zeros(len(mmetal))
        metal_outflow = fp_zeros(len(mmetal))

    try:
        dust_outflow = (mdust / mgas[0]) * gas_outflow * model_params["outflow_dust"]
    except KeyError:
        model_params["outflow_dust"] = fp_zeros(len(mdust))
        dust_outflow = fp_zeros(len(mdust))

    return gas_outflow, metal_outflow, dust_outflow
