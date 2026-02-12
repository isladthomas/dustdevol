from numpy import where, logspace, log10, vectorize, diff, searchsorted, clip
from dustdevol.adaptive.generic import fp, fp_zeros
from scipy.optimize import root


def life_from_mass_vec(masses, metallicity, metal_hist, gas_hist, stellar_lifetimes, t):
    """
    Function which finds stellar lifetime by solving the equation
    tau_f(Z(t - tau), m) - tau = 0
    for tau, taking into account the metallicity *at birth* for
    stars of mass m
    """

    tau0 = stellar_lifetimes((metallicity, masses))

    soln = root(
        lambda tau: stellar_lifetimes(
            ((metal_hist(t - tau) / gas_hist(t - tau))[:, 0], masses)
        )
        - tau,
        tau0,
    ).x

    return soln


def fast_supernova_rate(
    imf, metal_hist, gas_hist, sfr_hist, t, stellar_lifetimes, metallicity, cache
):
    """
    calculate rate of supernova events in SN / Gyr, assuming stars
    that go supernova have a short enough lifespan to be born and die
    in a single timestep (30-50 Myr)
    """

    try:
        masses = cache["sn_masses"]
        imf_vals = cache["sn_imf_values"]
        d_masses = cache["sn_d_masses"]
    except KeyError:
        masses = logspace(log10(8), log10(40), 257)
        d_masses = diff(masses)
        masses = masses[:-1] + (d_masses / 2)
        imf_vals = imf(masses)

        cache["sn_masses"] = masses
        cache["sn_imf_values"] = imf_vals
        cache["sn_d_masses"] = d_masses

    lifetimes = life_from_mass_vec(
        masses, metallicity, metal_hist, gas_hist, stellar_lifetimes, t
    )

    d_masses = where(t > lifetimes, d_masses, 0)

    sfr_vals = sfr_hist(t - lifetimes)

    sn_rate = (imf_vals * d_masses * sfr_vals).sum()

    return sn_rate
