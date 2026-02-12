import numpy as np
from astropy.cosmology import Planck13
from scipy import interpolate

# define working precision
fp = np.float64


# initialize an array of zeros with given shape, using wp
def fp_zeros(shape):
    return np.zeros(shape, dtype=fp)


# create a function which takes in time values and calculates
# the redshift at that time, assuming cosmological parameters
# from the Planck 2013 study
redshift_lookups = np.flip(np.concatenate(([0], np.logspace(-3, 3, 511))))
t_lookups = Planck13.age(redshift_lookups).value
z_at_t = interpolate.make_interp_spline(t_lookups, redshift_lookups, k=3)


# stand in for any of the gas/metal/dust evolution functions
# which just returns zeros, turning off that aspect of the model
def off(*args):
    return 0, 0, 0


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
    try:
        return cache["sfr_interp"]([t])[0]
    except KeyError:
        vals = np.loadtxt(model_params["sfr_file"], dtype=fp)
        vals *= [1e-9, 1e9]
        cache["sfr_interp"] = interpolate.CubicSpline(vals[:, 0], vals[:, 1])
        return cache["sfr_interp"]([t])[0]


# stellar lifetime table according to Schaller et. al 1992
# First column is initial mass in Msol, second is lifetime
# in Gyr at Z = 0.001 and third is lifetime at Z = 0.02
S92 = np.array(
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


S92S93_Z = np.array((0.001, 0.008, 0.02, 0.04))
S92S93_M = np.array(
    (0.8, 0.9, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0,
     7.0, 12.0, 20.0, 40.0, 60.0, 85.0, 120.0)
)

S92S93 = np.array(
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


stellar_lifetimes = interpolate.RegularGridInterpolator(
    (S92S93_Z, S92S93_M), S92S93 * 1e-9, method="cubic", bounds_error=False, fill_value=None
)


# SN dust production table according to Todini and Ferrara 2001
TF01 = np.array(
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
vdHG97_M92_yields = np.array(
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
        (np.nextafter(fp(40), np.inf), 0, 0, 0, 0, 6.4, 1.46, 6.4, 1.46),
        (60, 0, 0, 0, 0, 8.69, 1.03, 8.69, 1.03),
        (85, 0, 0, 0, 0, 17.75, 3.37, 17.75, 3.37),
        (120, 0, 0, 0, 0, 9.39, -0.13, 9.39, -0.13),
    ),
    dtype=fp,
)

# metallicity cutoffs for the previous yield table
# Z < 0.0025 means use the first set, z < 0.006 use the second, etc.
vdHG97_M92_cutoffs = np.array((0.0025, 0.006, 0.01, np.inf))

#
