from dustdevol.generic import (
    stellar_lifetimes_lin,
    h_stellar_lifetimes_lin,
)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.colors import hsv_to_rgb
import numpy as np


def tau_polinomyal_coefficients(z):
    """
    Coefficients (z-dependent) for the log(tau) formula from
    Raiteri C.M., Villata M. & Navarro J.F., 1996, A&A 315, 105-115

    """
    log_z = np.log10(z)
    log_z_2 = log_z**2

    a0 = 10.13 + 0.07547 * log_z - 0.008084 * log_z_2
    a1 = -4.424 - 0.7939 * log_z - 0.1187 * log_z_2
    a2 = 1.262 + 0.3385 * log_z + 0.05417 * log_z_2

    return [a0, a1, a2]


def stellar_lifetime(stellar_m, z):
    """
    Empirical formula for stellar lifetimes from
    Raiteri C.M., Villata M. & Navarro J.F., 1996, A&A 315, 105-115

    """
    log_m = np.log10(stellar_m)
    a0, a1, a2 = tau_polinomyal_coefficients(z)

    log_tau = a0 + a1 * log_m + a2 * (log_m**2)

    return np.pow(10, log_tau - 9)


masses = np.logspace(np.log10(0.8), np.log10(8), 500)
metallicities = np.linspace(0.001, 0.04, 2)
metallicity_hue = np.linspace(1.0, 0.75, 2)

hue_dict = {}
for z, hue in zip(metallicities, metallicity_hue):
    plt.semilogx(
        masses,
        (stellar_lifetimes_lin((z, masses)) - h_stellar_lifetimes_lin((z, masses)))
        / h_stellar_lifetimes_lin((z, masses)),
        alpha=0.2,
        color=hsv_to_rgb((hue, 1, 0.9)),
    )

plt.suptitle("Hydrogen- vs. Carbon-Burning Death Point")
plt.ylabel("Percent Difference (w.r.t. H-Burning)")
plt.xlabel("Initial Mass (Msol)")
plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1, 3))
plt.savefig("plots/direct_lifetime_comparison_percent.pdf", bbox_inches="tight")
plt.clf()

rescale = np.where(masses < 8, 1, 100)

for z, hue in zip(metallicities, metallicity_hue):
    plt.loglog(
        masses,
        stellar_lifetimes_lin((z, masses)) * rescale,
        alpha=1.0,
        color=hsv_to_rgb((hue, 1, 0.9)),
    )

    plt.loglog(
        masses,
        h_stellar_lifetimes_lin((z, masses)) * rescale,
        alpha=1.0,
        color=hsv_to_rgb((hue, 0.75, 0.75)),
    )

    plt.loglog(
        masses,
        stellar_lifetime(masses, z) * rescale,
        alpha=1.0,
        color=hsv_to_rgb((hue, 0.5, 0.5)),
    )

plt.suptitle("Hydrogen- vs. Carbon-Burning Death Point")
plt.ylabel("Lifetime (Gyr)")
plt.xlabel("Initial Mass (Msol)")
plt.axvline(x=8)
plt.savefig("plots/direct_lifetime_comparison.pdf", bbox_inches="tight")
plt.clf()
