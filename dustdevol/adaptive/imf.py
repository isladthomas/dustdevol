import numpy as np
import scipy.interpolate as interpolate
from scipy.integrate import quad as integrate


# helper function to allow for normalizing any of the below
# imfs for a different mass range than 0.1 to 100
def normalize_imf(imf, lower_m, upper_m):
    norm = integrate(lambda m: m * imf(m), lower_m, upper_m)

    def inner(m):
        return imf(m) / norm

    return inner


# NOTE: Double Check Things!
# IMF for the galactic disk and young clusters,
# proposed in Chabrier 2003, normalized so that
# int(m * imf) from 0.1 to 100 is 1
# though I think something's wrong, because units
# are diff too. Original Chab is number density / Msolar
# while this has units of 1 / Msolar^2
def bad_chab(m):
    if m <= 1.0:
        imf = np.exp(-1.0 * (np.log10(m) + 1.1023729)
                     * (np.log10(m) + 1.1023729))
        imf = (0.85 * imf) / 0.952199 / m
    else:
        imf = 0.24 * (m**-1.3) / m
    return imf


# version of the above function, except it interpolates
def interp_chab(lower, upper, n, k):
    masses = np.linspace(lower, upper, n)
    imfs = np.vectorize(chab)(masses)
    interp = interpolate.make_interp_spline(masses, imfs, k=k)
    return interp


# This is what I get by directly copying the chab function from Rowlands 2014
# and then normalizing to 1 from 0.1 to 120
def chab(m):
    if m <= 1.0:
        imf = 0.158 * \
            np.exp(-((np.log10(m) - np.log10(0.079)) ** 2) / (2 * 0.69**2))
    else:
        imf = 0.0443 * (m ** (-1.3))
    imf = imf / (m * np.log(10))
    return imf / 0.0815731452799614
