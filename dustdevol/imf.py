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


central_mass = fp(0.079)
sigma = fp(0.69)
A = exp(-(log10(central_mass) ** 2) / (2 * sigma**2))
lognorm_int = fp(0.206922920981)
M = A / log(10)


def generic_chab(m, power):
    imf = where(
        m <= 1.0,
        exp(-((log10(m) - log10(central_mass)) ** 2) / (2 * sigma**2)),
        A * (m ** (-power)),
    )
    imf = imf / (m * log(10))
    imf = imf / (lognorm_int + (M / (-power + 1)) * (120 ** (-power + 1) - 1))
    return imf


def chab(m):  # k = 1.47730718873342
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
    return generic_chab(m, 1.3)


def top_chab(m):  # k = 0.69150615262189
    """Disk IMF for single objects as defined by Chabrier 2003,
    normalized so ``m * chab(m)`` integrates to 1 over 0.1 to 120.
    Modified to be slightly more top-heavy.

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
    return generic_chab(m, 0.8)


def topper_chab(m):  # k = 0.3388691385116757
    """Disk IMF for single objects as defined by Chabrier 2003,
    normalized so ``m * chab(m)`` integrates to 1 over 0.1 to 120.
    Modified to be decently more top-heavy.

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
    return generic_chab(m, 0.5)


salp_norm = fp(5.8615127118)


def salp(m):  # k = 2.828956424049581
    """IMF for single objects as defined by Salpeter 1955,
    normalized so ``m*chab(m)`` integrates to 1 over 0.1 to 120.

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
    imf = m**-2.35
    imf = imf / salp_norm
    return imf


kroup_norm = fp(3.3376974089)


def kroup(m):  # k = 2.039436052588923
    """'Galactic Field' IMF defined by Kroupa & Weidner 2003
    normalized so ``m*chab(m)`` integrates to 1 over 0.1 to 120.

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
    imf = where(m <= 0.5, 2 * m**-1.3, m**-2.3)
    imf[m > 1] = (m**-2.7)[m > 1]
    imf = imf / kroup_norm
    return imf
