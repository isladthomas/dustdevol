import numpy as np


# NOTE: Double Check Things!
# IMF for the galactic disk and young clusters,
# proposed in Chabrier 2003, normalized so that
# int(m * imf) from 0.8 to 120 is 1
# though I think something's wrong, because units
# are diff too. Original Chab is number density / Msolar
# while this has units of 1 / Msolar^2
def chab(m):
    if m <= 1.0:
        imf = np.exp(-1.0 * (np.log10(m) + 1.1023729)
                     * (np.log10(m) + 1.1023729))
        imf = (0.85 * imf) / 0.952199 / m
    else:
        imf = 0.24 * (m**-1.3) / m
    return imf


# This is what I get by directly copying the chab function and then
# normalizing to 1 from 0.8 to 120
def alt_chab(m):
    if m <= 1.0:
        imf = 0.158 * \
            np.exp(-((np.log10(m) - np.log10(0.079)) ** 2) / (2 * 0.69**2))
    else:
        imf = 0.044 * (m ** (-1.3))
    imf = imf / (m * np.log(10))
    return imf / 0.08039084139276075
