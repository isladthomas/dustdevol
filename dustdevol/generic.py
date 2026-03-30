from numpy import (
    float64,
    zeros,
    array,
    empty,
    full,
    flip,
    concatenate,
    logspace,
    loadtxt,
    nextafter,
    inf,
    clip,
)
from astropy.cosmology import Planck13
from scipy.interpolate import make_interp_spline, RegularGridInterpolator
from scipy.stats import skewnorm

# define working precision
# NOTE: Due to the way scipy.interpolate.PPoly works, there's no point setting
# this other than np.float64. The interpolate functions always use np.float64
# internally, and will not be sped up by switching this to float16 or float32
# and will actually throw an error if set to np.float128, as the extend()
# method will change the internal rep to float128, and it will errstop
# the next time it is evaluated.
fp = float64


def fp_zeros(shape):
    """Returns an array of all zeros with the given shape in dustdevol's
    working precision.

    Parameters
    ----------
    shape : int or tuple of ints
            Shape of the created array.

    Returns
    -------
    out : ndarray
          Zero array with given shape using working precision.
    """
    return zeros(shape, dtype=fp)


def fp_array(iterable):
    """Returns the given iterable as a numpy array using dustdevol's working
    precision.

    Parameters
    ----------
    iterable : array_like
               Object to be turned into an array.


    Returns
    -------
    out : ndarray
          Array with given values using working precision.
    """
    return array(iterable, dtype=fp)


# initialize an empty array with given shape, using wp
def fp_empty(shape):
    """Returns an uninitialized array with the given shape in dustdevol's
    working precision.

    Parameters
    ----------
    shape : int or tuple of ints
            Shape of the created array.

    Returns
    -------
    out : ndarray
          Empty array with given shape using working precision.
    """
    return empty(shape, dtype=fp)


# initialize an empty array with given shape, using wp
def fp_full(shape, value):
    """Returns an array filled with the given values with the given shape
    in dustdevol's working precision.

    Parameters
    ----------
    shape : int or tuple of ints
            Shape of the created array.
    value : float
            Value to fill the array with.

    Returns
    -------
    out : ndarray
          Filled array with given shape using working precision.
    """
    return full(shape, value, dtype=fp)


# create a function which takes in time values and calculates
# the redshift at that time, assuming cosmological parameters
# from the Planck 2013 study
redshift_lookups = flip(concatenate(([fp(0)], logspace(-3, 3, 511, dtype=fp))))
t_lookups = fp_array(Planck13.age(redshift_lookups).value)
z_at_t = make_interp_spline(t_lookups, redshift_lookups, k=3)


# stand in for any of the gas/metal/dust evolution functions
# which just returns zeros, turning off that aspect of the model
def off(*args):
    """stand in for any gas/metal/dust evolution functions, returns all zeros
    turning off that component.

    Params
    ------
    args : Arbitrary

    Returns
    -------
    vals : float, float, float
           zeros
    """
    return fp(0), fp(0), fp(0)


# reads sfh from a file, interpolating using either a user specified
# order, or defaulting to a cubic interpolation.
# file should consist of a series of lines starting with time in yrs,
# a space, and then sfr in Msol/yr
def sfr_from_file(
    model_params,
    t,
    redshift,
    mgas,
    mstar,
    mmetal,
    mdust,
    gas_hist,
    star_hist,
    metal_hist,
    dust_hist,
    cache,
):
    """Function which reads sfh from a file, with each line containing
    time in years, a space, and then sfr in Msol/yr at that time.
    Reads the filename and order of interpolation between times from
    model_params.

    Parameters
    ----------
    model_params : dict
                  Must contain "sfr_file", with the associated key being the
                  filename of the .sfh file. May also contain
                  "sfr_interp_order", which gives the order of interpolation,
                  will default to 3 (cubic) if absent.

    Returns
    -------
    out : float
          sfr in Msol/Gyr.
    """

    try:
        return cache["sfr_interp"]([t])[0]
    except KeyError:
        vals = loadtxt(model_params["sfr_file"], dtype=fp)
        vals *= fp_array([1e-9, 1e9])
        try:
            cache["sfr_interp"] = make_interp_spline(
                vals[:, 0], vals[:, 1], k=model_params["sfr_interp_order"]
            )
        except KeyError:
            cache["sfr_interp"] = make_interp_spline(
                vals[:, 0], vals[:, 1], k=3)
        return cache["sfr_interp"]([t])[0]


# Type Ia DTD from Strolger
unnormed_S2020 = skewnorm(220, loc=0.01, scale=0.6).pdf


def S2020(t):
    return unnormed_S2020(t) / 0.960771865064329


# stellar lifetime table according to Schaller et. al 1992
# First column is initial mass in Msol, second is lifetime
# in Gyr at Z = 0.001 and third is lifetime at Z = 0.02
S92 = fp_array(
    (
        (0.8, 15.0, 26.0),
        (0.9, 9.5, 15.0),
        (1.0, 6.3, 10.0),
        (1.5, 1.8, 2.7),
        (2.0, 0.86, 1.1),
        (3.0, 0.29, 0.35),
        (4.0, 0.14, 0.16),
        (5.0, 0.088, 0.094),
        (7.0, 0.045, 0.043),
        (9.0, 0.029, 0.026),
        (12.0, 0.018, 0.016),
        (20.0, 0.0094, 0.0081),
        (40.0, 0.0049, 0.0043),
        (60.0, 0.0037, 0.0034),
        (85.0, 0.0031, 0.0028),
        (120.0, 0.0028, 0.0026),
    )
)


# stellar lifetime metallicities, masses, and grid of values according to
# Schaller et. al. 1992 and Schaerer et. al. 1993, then interpolated
# cubically
S92S93_Z = fp_array((0.001, 0.008, 0.02, 0.04))
S92S93_M = fp_array(
    (0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
     7.0, 12.0, 20.0, 40.0, 60.0, 85.0, 120.0)
)

S92S93 = fp_array(
    (
        (
            1.524235e10,
            1.072937e10,
            7.285024e09,
            1.986736e09,
            1.027951e09,
            3.413019e08,
            1.658614e08,
            1.002186e08,
            5.013853e07,
            1.980556e07,
            1.024804e07,
            5.340652e06,
            4.081399e06,
            3.384446e06,
            3.071335e06,
        ),
        (
            2.361184e10,
            1.511880e10,
            1.013239e10,
            2.434480e09,
            1.232497e09,
            3.996536e08,
            1.830133e08,
            1.089796e08,
            5.183734e07,
            1.956258e07,
            9.954179e06,
            5.259286e06,
            4.106728e06,
            3.478416e06,
            3.107879e06,
        ),
        (
            2.502788e10,
            1.744009e10,
            1.226981e10,
            2.910760e09,
            1.411247e09,
            4.405361e08,
            1.942839e08,
            1.084538e08,
            4.839158e07,
            1.767493e07,
            8.964268e06,
            4.941880e06,
            4.172452e06,
            3.434571e06,
            3.519620e06,
        ),
        (
            2.710062e10,
            1.729064e10,
            1.209014e10,
            2.795810e09,
            1.106791e09,
            4.222424e08,
            1.782498e08,
            9.707679e07,
            4.233631e07,
            1.501992e07,
            7.712726e06,
            4.332250e06,
            3.530899e06,
            3.132593e06,
            2.919864e06,
        ),
    )
)

S92S93_H_burning = fp_array(
    (
        (
            1.502953e10,
            9.452348e09,
            6.263588e09,
            1.842735e09,
            8.556338e08,
            2.907975e08,
            1.440681e08,
            8.827634e07,
            4.505329e07,
            1.811635e07,
            9.384113e06,
            4.890901e06,
            3.714768e06,
            3.062983e06,
            2.779810e06,
        ),
        (
            2.080469e10,
            1.284153e10,
            8.235291e09,
            2.236135e09,
            9.656467e08,
            3.261578e08,
            1.575337e08,
            9.559587e07,
            4.647998e07,
            1.783118e07,
            9.109045e06,
            4.810642e06,
            3.704545e06,
            3.123648e06,
            2.779076e06,
        ),
        (
            2.502788e10,
            1.550030e10,
            9.844567e09,
            2.694651e09,
            1.115943e09,
            3.525031e08,
            1.647344e08,
            9.445912e07,
            4.318800e07,
            1.601758e07,
            8.140933e06,
            4.303191e06,
            3.446927e06,
            2.822791e06,
            2.561435e06,
        ),
        (
            2.433909e10,
            1.500648e10,
            9.514924e09,
            2.591387e09,
            1.056634e09,
            3.253642e08,
            1.451816e08,
            8.228334e07,
            3.715984e07,
            1.344850e07,
            6.924528e06,
            3.806840e06,
            3.026151e06,
            2.651363e06,
            2.433512e06,
        ),
    )
)


stellar_lifetimes = RegularGridInterpolator(
    (S92S93_Z, S92S93_M),
    S92S93 * 1e-9,
    method="cubic",
    bounds_error=False,
    fill_value=None,
)
stellar_lifetimes_pchip = RegularGridInterpolator(
    (S92S93_Z, S92S93_M),
    S92S93 * 1e-9,
    method="pchip",
    bounds_error=False,
    fill_value=None,
)
stellar_lifetimes_lin = RegularGridInterpolator(
    (S92S93_Z, S92S93_M),
    S92S93 * 1e-9,
    method="linear",
    bounds_error=False,
    fill_value=None,
)
stellar_lifetimes_nn = RegularGridInterpolator(
    (S92S93_Z, S92S93_M),
    S92S93 * 1e-9,
    method="nearest",
    bounds_error=False,
    fill_value=None,
)

h_stellar_lifetimes = RegularGridInterpolator(
    (S92S93_Z, S92S93_M),
    S92S93_H_burning * 1e-9,
    method="cubic",
    bounds_error=False,
    fill_value=None,
)
h_stellar_lifetimes_pchip = RegularGridInterpolator(
    (S92S93_Z, S92S93_M),
    S92S93_H_burning * 1e-9,
    method="pchip",
    bounds_error=False,
    fill_value=None,
)
h_stellar_lifetimes_lin = RegularGridInterpolator(
    (S92S93_Z, S92S93_M),
    S92S93_H_burning * 1e-9,
    method="linear",
    bounds_error=False,
    fill_value=None,
)
h_stellar_lifetimes_nn = RegularGridInterpolator(
    (S92S93_Z, S92S93_M),
    S92S93_H_burning * 1e-9,
    method="nearest",
    bounds_error=False,
    fill_value=None,
)


# SN dust production table according to Todini and Ferrara 2001
TF01 = fp_array(
    (
        (8.5, 0),
        (9, 0.7674),
        (12, 0.5),
        (15, 0.3508),
        (20, 0.176),
        (25, 0.2239),
        (30, 0.15),
        (35, 0.08),
        (40, 0.0451),
    )
)

# van den Hoek and Maeder metal yields table.
# first column is initial mass, subsequent columns are
# total metal yield and oxygen yield at Z = 0.001, 0.004, 0.008, 0.02
# the np.nextafter line is to ensure that, when nn interpolating,
# only yields from stellar winds are considered for m > 40
# as what would be considered "supernova ejecta," while still
# calculable for these stars, will be trapped in a black hole
vdHG97_M92_yields = fp_array(
    (
        (0.9, 0, -1.773e-06, 9.72e-06, -6.498e-07,
         6.147e-05, 2.565e-05, 0, -3.483e-05),
        (1.0, 0, -2.23e-06, 0.000854, 6.36e-05,
         0.000112, 5.36e-05, 0.00161, 0.000981),
        (
            1.3,
            0.004017,
            0.0003237,
            0.002587,
            0.0001872,
            0.00221,
            0.0001807,
            0.003939,
            0.002431,
        ),
        (
            1.5,
            0.005295,
            0.000426,
            0.003855,
            0.0003105,
            0.00459,
            0.000324,
            0.00312,
            0.0005565,
        ),
        (
            1.7,
            0.006409,
            0.0005168,
            0.006783,
            0.0005353,
            0.005967,
            0.0004233,
            0.004029,
            0.0003026,
        ),
        (
            2.0,
            0.01058,
            0.0008,
            0.01172,
            0.000722,
            0.01086,
            0.000508,
            0.00788,
            0.0001554,
        ),
        (
            2.5,
            0.013975,
            0.0010725,
            0.01705,
            0.00099,
            0.01645,
            0.0005875,
            0.014,
            -0.00010475,
        ),
        (
            3.0,
            0.01605,
            0.001251,
            0.02529,
            0.001536,
            0.02394,
            0.000915,
            0.02076,
            -4.41e-06,
        ),
        (
            4.0,
            0.02624,
            0.001724,
            0.02349,
            0.001104,
            0.02176,
            0.0002288,
            0.02496,
            -0.000864,
        ),
        (5.0, 0.0386, 0.00206, 0.03535, 0.001285,
         0.03295, 0.00033, 0.0314, -0.001455),
        (
            7.0,
            0.06727,
            0.0004403,
            0.06216,
            -0.001778,
            0.05845,
            -0.004424,
            0.05418,
            -0.00896,
        ),
        (
            8.0,
            0.07024,
            0.000768,
            0.07656,
            -0.002568,
            0.0816,
            -0.007936,
            0.06728,
            -0.0128,
        ),
        (9, 0.27, 0.004, 0.27, 0.004, 0.173, 0, 0.173, 0),
        (12, 0.83, 0.15, 0.83, 0.15, 0.686, 0.11, 0.686, 0.11),
        (15, 1.53, 0.46, 1.53, 0.46, 1.32, 0.41, 1.32, 0.41),
        (20, 2.93, 1.27, 2.93, 1.27, 2.73, 1.27, 2.73, 1.27),
        (25, 4.45, 2.40, 4.45, 2.40, 4.48, 2.57, 4.48, 2.57),
        (40, 9.71, 6.80, 9.71, 6.80, 8.01, 2.08, 8.01, 2.08),
        (nextafter(fp(40), inf), 0, 0, 0, 0, 6.4, 1.46, 6.4, 1.46),
        (60, 0, 0, 0, 0, 8.69, 1.03, 8.69, 1.03),
        (85, 0, 0, 0, 0, 17.75, 3.37, 17.75, 3.37),
        (120, 0, 0, 0, 0, 9.39, -0.13, 9.39, -0.13),
    ),
)

# metallicity cutoffs for the previous yield table
# Z < 0.0025 means use the first set, z < 0.006 use the second, etc.
vdHG97_M92_cutoffs = fp_array((0.0025, 0.006, 0.01, inf))

# yield tables for type Ia SN, from Table 3 in Leung & Nomoto 2018
# note that we have two identical columns. This is because, interanally,
# the metallicity lookup function doesn't work with only 1 initial mass value
LN2018_yields = fp_array(
    [
        [1.31249031e00, 4.19000000e-02, 7.66300000e-07],
        [1.39350736e00, 4.45000000e-02, 6.59400000e-09],
        [1.31688755e00, 5.38000000e-02, 1.31400000e-09],
        [1.31471865e00, 5.45000002e-02, 8.00000000e-10],
        [1.39674679e00, 5.49000005e-02, 1.19560000e-09],
        [1.33574176e00, 5.49000001e-02, 2.45000000e-10],
        [1.36319016e00, 6.55000005e-02, 7.11200000e-10],
    ]
)

LN2018_Z = fp_array((0, 0.00134, 0.0067, 0.0134, 0.0268, 0.0402, 0.067))

LN2018_nn = make_interp_spline(LN2018_Z, LN2018_yields, k=0)
LN2018_lin_extrap = make_interp_spline(LN2018_Z, LN2018_yields, k=1)


def LN2018_lin(z):
    z = clip(z, 0, 0.067)
    return LN2018_lin_extrap(z)


KA18_high_Z = fp_array((0.03, 0.014, 0.007, 0.0028, 0.001))
KA18_high_M = fp_array(
    (
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.25,
        3.5,
        3.75,
        4.0,
        4.5,
        5.0,
        5.5,
        6.0,
        7.0,
    )
)

KA18_high_yields = fp_array(
    [
        [
            [1.22671527e-02, 5.24461000e-03, 9.62580000e-04],
            [1.52109576e-02, 6.49090400e-03, 1.46856800e-03],
            [1.71790445e-02, 7.31960000e-03, 1.84853333e-03],
            [1.85725999e-02, 7.90360000e-03, 2.14756000e-03],
            [1.95757703e-02, 8.34550000e-03, 2.02671000e-03],
            [2.03701213e-02, 8.68471111e-03, 2.08057333e-03],
            [2.73882232e-02, 9.01708000e-03, 2.34128400e-03],
            [2.78496288e-02, 9.00850909e-03, 2.71101818e-03],
            [2.75065261e-02, 9.06996667e-03, 2.84669333e-03],
            [2.76635536e-02, 9.12184615e-03, 2.90075385e-03],
            [2.97384088e-02, 9.12331429e-03, 3.07560000e-03],
            [3.01403074e-02, 9.23754667e-03, 3.09568000e-03],
            [2.79716552e-02, 9.31027500e-03, 3.31692500e-03],
            [2.55656360e-02, 9.48008889e-03, 3.34400000e-03],
            [2.64666363e-02, 9.31766000e-03, 3.82066000e-03],
            [2.68621527e-02, 9.24550909e-03, 5.25860000e-03],
            [2.72619751e-02, 9.16528333e-03, 7.31690000e-03],
            [2.69555400e-02, 8.86205714e-03, 8.39644286e-03],
        ],
        [
            [5.92248005e-03, 2.51199000e-03, 4.74836000e-04],
            [7.38485289e-03, 3.12602400e-03, 7.29140800e-04],
            [1.05183049e-02, 3.67158000e-03, 9.49140000e-04],
            [1.15067368e-02, 3.95117143e-03, 1.08040571e-03],
            [1.24445353e-02, 4.14306500e-03, 1.11697500e-03],
            [1.30817643e-02, 4.25728889e-03, 1.18762222e-03],
            [1.48011835e-02, 4.36480000e-03, 1.34511200e-03],
            [1.67505595e-02, 4.43701818e-03, 1.44028364e-03],
            [1.80708432e-02, 4.46316667e-03, 1.55627667e-03],
            [1.91323553e-02, 4.49261538e-03, 1.63134462e-03],
            [1.85769344e-02, 4.52917143e-03, 1.63377714e-03],
            [1.78580652e-02, 4.54234667e-03, 1.70907200e-03],
            [1.64950687e-02, 4.59202500e-03, 1.69759500e-03],
            [1.72638902e-02, 4.64553333e-03, 4.82373333e-03],
            [1.80668002e-02, 4.43592000e-03, 7.79418000e-03],
            [1.80589505e-02, 4.07336364e-03, 8.40072727e-03],
            [1.71773575e-02, 3.73655000e-03, 7.99943333e-03],
            [1.51363146e-02, 3.52052857e-03, 6.58417143e-03],
        ],
        [
            [2.81311134e-03, 1.18895000e-03, 2.41779000e-04],
            [3.55745939e-03, 1.50265600e-03, 3.69484800e-04],
            [5.29481920e-03, 1.78586000e-03, 4.64840000e-04],
            [7.58487884e-03, 2.05188000e-03, 5.52564000e-04],
            [1.19996354e-02, 2.56697500e-03, 7.15205000e-04],
            [1.40496212e-02, 2.56488444e-03, 8.20862222e-04],
            [1.64793697e-02, 2.55987200e-03, 8.53932000e-04],
            [1.65509108e-02, 2.53964000e-03, 8.88970909e-04],
            [1.54119202e-02, 2.48060000e-03, 9.10823333e-04],
            [1.42030367e-02, 2.51551692e-03, 9.00649231e-04],
            [1.27932551e-02, 2.55981429e-03, 8.64771429e-04],
            [1.15450573e-02, 2.56202133e-03, 9.06826667e-04],
            [1.82782632e-02, 2.36508500e-03, 1.19717000e-02],
            [1.71164194e-02, 1.92190444e-03, 1.12874222e-02],
            [1.51486177e-02, 1.58745000e-03, 1.00899600e-02],
            [1.18536798e-02, 1.38191091e-03, 7.29761818e-03],
            [9.41118072e-03, 1.59628333e-03, 4.74426667e-03],
            [0.00000000e00, 0.00000000e00, 0.00000000e00],
        ],
        [
            [4.69273235e-06, -5.70281000e-06, 4.44708000e-05],
            [1.12168771e-03, 8.08136000e-05, 8.18520000e-05],
            [2.20436571e-03, 1.37732000e-04, 1.17556667e-04],
            [4.26986219e-03, 1.53969714e-04, 1.43996000e-04],
            [7.47009237e-03, 3.51816000e-04, 1.59743500e-04],
            [9.50025713e-03, 2.45492444e-04, 1.91577333e-04],
            [1.12894600e-02, 3.04843200e-04, 2.31293600e-04],
            [1.08054246e-02, 1.93560000e-04, 1.96800364e-04],
            [8.91838803e-03, 1.83484333e-04, 2.28986667e-04],
            [6.61999049e-03, 1.22985846e-04, 2.02833846e-04],
            [5.54482949e-03, 1.24589714e-04, 2.45348571e-04],
            [4.32113601e-03, 8.66008000e-05, 2.56793600e-04],
            [4.68816420e-03, 8.77470000e-05, 3.94870000e-04],
            [4.90117503e-03, -7.07577778e-05, 4.23315556e-03],
            [3.96994928e-03, -2.35570000e-04, 3.90428000e-03],
            [3.03870587e-03, -4.13158182e-04, 3.41958182e-03],
            [2.10916370e-03, -5.33398333e-04, 2.75923333e-03],
            [7.36063993e-04, -5.95547143e-04, 1.60085714e-03],
        ],
        [
            [3.24884439e-04, 1.39775400e-04, 1.68796900e-05],
            [4.69239371e-04, 2.01881360e-04, 2.43797840e-05],
            [5.55525193e-04, 2.39004200e-04, 2.88628467e-05],
            [0.00000000e00, 0.00000000e00, 0.00000000e00],
            [6.53578872e-04, 2.81189900e-04, 3.39573250e-05],
            [6.84354126e-04, 2.94430400e-04, 3.55562844e-05],
            [6.99328199e-04, 3.00872720e-04, 3.63342760e-05],
            [7.11512740e-04, 3.06114873e-04, 3.69673455e-05],
            [7.18329519e-04, 3.09047667e-04, 3.73215000e-05],
            [7.22997570e-04, 3.11056000e-04, 3.75640308e-05],
            [7.37170537e-04, 3.17153714e-04, 3.83004000e-05],
            [0.00000000e00, 0.00000000e00, 0.00000000e00],
            [7.60659191e-04, 3.27259250e-04, 3.95207750e-05],
            [7.79168525e-04, 3.35222444e-04, 4.04824667e-05],
            [7.93005111e-04, 3.41175400e-04, 4.12013600e-05],
            [8.03697081e-04, 3.45775455e-04, 4.17568545e-05],
            [8.10972328e-04, 3.48905500e-04, 4.21348500e-05],
            [8.16551518e-04, 3.51305857e-04, 4.24247286e-05],
        ],
    ]
)

KA18_high_nn_extrap = RegularGridInterpolator(
    (KA18_high_Z, KA18_high_M),
    KA18_high_yields,
    method="nearest",
    bounds_error=False,
    fill_value=None,
)
KA18_high_lin_extrap = RegularGridInterpolator(
    (KA18_high_Z, KA18_high_M),
    KA18_high_yields,
    method="linear",
    bounds_error=False,
    fill_value=None,
)


def KA18_high_nn(z, m):
    z = clip(z, 0.001, 0.03)
    m = clip(m, 1, 7)
    return KA18_high_nn_extrap((z, m))


def KA18_high_lin(z, m):
    z = clip(z, 0.001, 0.03)
    m = clip(m, 1, 7)
    return KA18_high_lin_extrap((z, m))


LC18_R150_Z = fp_array((0.0134, 0.00134, 0.000134, 0.0000134))
LC18_R150_M = fp_array((13.0, 15.0, 20.0, 25.0, 30.0, 40.0, 60.0, 80.0, 120.0))

LC18_R150_yields = fp_array(
    [
        [
            [1.48801763e-01, 9.63230769e-02, 2.44600000e-03],
            [1.90664192e-01, 1.23560000e-01, 2.15233333e-03],
            [2.12339711e-01, 1.25980000e-01, 2.80580000e-03],
            [2.39904794e-01, 1.40564000e-01, 2.82524000e-03],
            [1.21172303e-02, -1.25516667e-04, 3.14250000e-03],
            [1.47630089e-02, -3.38125000e-04, 3.57700000e-03],
            [2.13673875e-02, -1.04245000e-04, 3.56083333e-03],
            [2.49780574e-02, 3.95437500e-04, 3.70162500e-03],
            [1.87330029e-02, -1.61275000e-03, 4.21608333e-03],
        ],
        [
            [1.66784348e-01, 9.52769231e-02, 5.36284615e-03],
            [1.90165412e-01, 1.25680000e-01, 6.34893333e-03],
            [2.36729868e-01, 1.66100000e-01, 6.78400000e-03],
            [2.74887269e-01, 1.94440000e-01, 3.34536000e-03],
            [-8.35675654e-06, -1.06793333e-04, 1.26753333e-04],
            [-4.85350752e-05, -6.23250000e-04, 7.43625000e-04],
            [1.89206298e-02, 1.51310000e-03, 7.56016667e-04],
            [2.68341807e-02, 3.18000000e-03, 6.88725000e-04],
            [1.79851301e-02, 1.09866667e-03, 9.71500000e-04],
        ],
        [
            [1.69824594e-01, 1.13638462e-01, 4.64176923e-03],
            [1.89900886e-01, 1.10700000e-01, 4.66853333e-03],
            [2.16928476e-01, 1.55125000e-01, 5.04300000e-03],
            [2.59452615e-01, 1.85400000e-01, 3.25248000e-04],
            [-4.29269547e-07, -9.17933333e-06, 1.26483333e-05],
            [8.01477910e-04, 8.99300000e-05, 4.26400000e-04],
            [1.47675572e-03, 3.61633333e-04, 4.70216667e-04],
            [2.86152360e-03, 7.99737500e-04, 5.38825000e-04],
            [6.87829265e-03, 1.92400000e-03, 5.14125000e-04],
        ],
        [
            [1.58404404e-01, 1.12238462e-01, 2.58723077e-03],
            [1.96795954e-01, 1.34146667e-01, 1.17960000e-03],
            [2.14175228e-01, 1.56285000e-01, 2.64885000e-03],
            [2.58298363e-01, 1.88192000e-01, 3.46280000e-03],
            [3.02827262e-06, -1.03210000e-07, 3.11766667e-06],
            [-5.63156570e-08, -2.56950000e-06, 3.99075000e-06],
            [-6.08343996e-08, -2.09933333e-06, 3.13616667e-06],
            [-1.43481428e-07, -2.95787500e-06, 3.97650000e-06],
            [5.31493244e-06, -4.96366667e-06, 1.14000000e-05],
        ],
    ]
)

LC18_R150_nn_extrap = RegularGridInterpolator(
    (LC18_R150_Z, LC18_R150_M),
    LC18_R150_yields,
    method="nearest",
    bounds_error=False,
    fill_value=None,
)
LC18_R150_lin_extrap = RegularGridInterpolator(
    (LC18_R150_Z, LC18_R150_M),
    LC18_R150_yields,
    method="linear",
    bounds_error=False,
    fill_value=None,
)


def LC18_R150_nn(z, m):
    z = clip(z, 0.0000134, 0.0134)
    m = clip(m, 13, 120)
    return LC18_R150_nn_extrap((z, m))


def LC18_R150_lin(z, m):
    z = clip(z, 0.0000134, 0.0134)
    m = clip(m, 13, 120)
    return LC18_R150_lin_extrap((z, m))


def AGB_SN_yields(AGB_interp, SN_interp, iso_num):
    def inner(z, m):
        soln = fp_empty((m.shape[0], iso_num))
        soln[m <= 8] = AGB_interp(z[m <= 8], m[m <= 8])
        soln[m > 8] = SN_interp(z[m > 8], m[m > 8])
        soln = soln * m[:, None]
        return soln

    return inner
