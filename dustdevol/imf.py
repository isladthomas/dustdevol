from numpy import exp, log10, where, log
from dustdevol.generic import fp


def BEDE_chab(m):
    """Chabrier IMF used in De Vis 2017 and BEDE (De Vis 2020),
    normalized so ``m * chab(m)`` integrates to 1 over 0.1 to 100,
    and erroneously missing the sigma term in the log-normal region

    Parameters
    ----------
        m : (m,)
            ndarray of masses

    Returns
    -------
        vals : (m,)
               ndarray corresponding to values of the imf at each of the
               given masses.
    """
    if m <= 1.0:
        imf = exp(-1.0 * (log10(m) + 1.1023729) * (log10(m) + 1.1023729))
        imf = (0.85 * imf) / 0.952199 / m
    else:
        imf = 0.24 * (m**-1.3) / m
    return imf


a_exp = fp(0.158)
central_mass = fp(0.079)
sigma = fp(0.69)

a_pow = fp(0.0443)
pow = fp(1.3)

norm = fp(0.0815731452799614)


# This is what I get by directly copying the chab function from Rowlands 2014
# and then normalizing to 1 from 0.1 to 120
def chab(m):
    """Disk IMF for single objects as defined by Chabrier 2003,
    normalized so ``m * chab(m)`` integrates to 1 over 0.1 to 120.

    Parameters
    ----------
        m : (m,)
            ndarray of masses

    Returns
    -------
        out : (m,)
               ndarray corresponding to values of the imf at each of the
               given masses.
    """

    imf = where(
        m <= 1.0,
        a_exp * exp(-((log10(m) - log10(central_mass)) ** 2) / (2 * sigma**2)),
        a_pow * (m ** (-pow)),
    )
    imf = imf / (m * log(10))
    return imf / norm
