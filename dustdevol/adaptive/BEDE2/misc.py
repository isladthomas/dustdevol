from numpy import where, logspace, log10, vectorize, diff, searchsorted, clip
from dustdevol.adaptive.generic import fp, fp_zeros
from scipy.optimize import root


def life_from_mass_vec(masses, metal_hist, gas_hist, stellar_lifetimes, t, cache):
    """
    Function which finds stellar lifetime by solving the equation
    tau_f(Z(t - tau), m) - tau = 0
    for tau, taking into account the metallicity *at birth* for
    stars of mass m
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
