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
z_at_t = interpolate.make_interp_spline(t_lookups, redshift_lookups, k=4)


# stand in for any of the gas/metal/dust evolution functions
# which just returns zeros, turning off that aspect of the model
def off(*args):
    gas = fp_zeros(len(args[-4]))
    metal = fp_zeros(len(args[-2]))
    dust = fp_zeros(len(args[-1]))
    return gas, metal, dust


# reads sfh from a file, interpolating using either a user specified
# order, or defaulting to a cubic interpolation.
# file should consist of a series of lines starting with time in yrs,
# a space, and then sfr in Msol/yr
def sfr_from_file(model_params, times):
    vals = np.loadtxt(model_params["sfr_file"], dtype=fp)
    vals *= [1e-9, 1e9]
    if (np.shape(times) == np.shape(vals[:, 0])) and (times == vals[:, 0]):
        return vals[:, 1]
    else:
        try:
            interp = interpolate.make_interp_spline(
                vals[:, 0], vals[:, 1], k=model_params["sfr_interp_order"]
            )
        except KeyError:
            print("No sfr interpolation specified, defaulting to order 3")
            interp = interpolate.make_interp_spline(
                vals[:, 0], vals[:, 1], k=3)
        return interp(times)


# stellar lifetime table according to INSERT
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


# SN dust production table according to INSERT
TF01 = np.array(
    (
        (8.5, 0),
        (9, 0.17),
        (12, 0.2),
        (15, 0.5),
        (20, 0.5),
        (22, 0.8),
        (25, 1.0),
        (30, 1.0),
        (35, 0.6),
        (40, 0.4),
    )
)


vdHG97_M92_metals_raw = np.array(
    (
        (0.9, 0, 0.0, 0, 9.72e-06, 0, 6.147e-05, 0, 0.0),
        (1.0, 0, 0.0, 0, 0.000854, 0, 0.000112, 0, 0.00161),
        (1.3, 0, 0.004017, 0, 0.002587, 0, 0.00221, 0, 0.003939),
        (1.5, 0, 0.005295, 0, 0.003855, 0, 0.00459, 0, 0.00312),
        (1.7, 0, 0.006409, 0, 0.006783, 0, 0.005967, 0, 0.004029),
        (2.0, 0, 0.01058, 0, 0.01172, 0, 0.01086, 0, 0.00788),
        (2.5, 0, 0.013975, 0, 0.01705, 0, 0.01645, 0, 0.014),
        (3.0, 0, 0.01605, 0, 0.02529, 0, 0.02394, 0, 0.02076),
        (4.0, 0, 0.02624, 0, 0.02348, 0, 0.02176, 0, 0.02496),
        (5.0, 0, 0.0386, 0, 0.03535, 0, 0.03295, 0, 0.0314),
        (7.0, 0, 0.06727, 0, 0.06216, 0, 0.05845, 0, 0.05418),
        (8.0, 0, 0.07024, 0, 0.07656, 0, 0.0816, 0, 0.06728),
        (9, 0.27, 0, 0.27, 0, 0.173, 0, 0.173, 0),
        (12, 0.83, 0, 0.83, 0, 0.686, 0, 0.686, 0),
        (15, 1.53, 0, 1.53, 0, 1.32, 0, 1.32, 0),
        (20, 2.93, 0, 2.93, 0, 2.73, 0, 2.73, 0),
        (25, 4.45, 0, 4.45, 0, 4.48, 0, 4.48, 0),
        (40, 9.71, 0, 9.71, 0, 1.61, 6.4, 1.61, 6.4),
        (60, 17.1, 0, 17.1, 0, 1.16, 8.69, 1.16, 8.69),
        (85, 26.7, 0, 26.7, 0, 1.56, 17.75, 1.56, 17.75),
        (120, 41.6, 0, 41.6, 0, 0.72, 9.39, 0.72, 9.39),
    )
)


vdHG97_M92_oxy_raw = np.array(
    (
        (0.9, 0, -1.773e-06, 0, -6.498e-07, 0, 2.565e-05, 0, -3.483e-05),
        (1, 0, -2.23e-06, 0, 6.36e-05, 0, 5.36e-05, 0, 0.000981),
        (1.3, 0, 0.0003237, 0, 0.0001872, 0, 0.0001807, 0, 0.002431),
        (1.5, 0, 0.000426, 0, 0.0003105, 0, 0.000324, 0, 0.0005565),
        (1.7, 0, 0.0005168, 0, 0.0005253, 0, 0.0004233, 0, 0.0003026),
        (2, 0, 0.0008, 0, 0.000722, 0, 0.000508, 0, 0.0001554),
        (2.5, 0, 0.0010725, 0, 0.00099, 0, 0.0005875, 0, -0.00010475),
        (3, 0, 0.001251, 0, 0.001536, 0, 0.000915, 0, -4.41e-06),
        (4, 0, 0.001724, 0, 0.001104, 0, 0.0002288, 0, -0.000864),
        (5, 0, 0.00206, 0, 0.001285, 0, 0.00033, 0, -0.001455),
        (7, 0, 0.0004403, 0, -0.001778, 0, -0.004424, 0, -0.00896),
        (8, 0, 0.000768, 0, -0.002568, 0, -0.007936, 0, -0.0128),
        (9, 0.004, 0, 0.004, 0, 0, 0, 0, 0),
        (12, 0.15, 0, 0.15, 0, 0.11, 0, 0.11, 0),
        (15, 0.46, 0, 0.46, 0, 0.41, 0, 0.41, 0),
        (20, 1.27, 0, 1.27, 0, 1.27, 0, 1.27, 0),
        (25, 2.40, 0, 2.40, 0, 2.60, -0.03, 2.60, -0.03),
        (40, 6.80, 0, 6.80, 0, 0.62, 1.46, 0.62, 1.46),
        (60, 14.20, 0, 14.20, 0, 0.40, 1.03, 0.40, 1.03),
        (85, 22.60, 0, 22.60, 0, 0.59, 3.37, 0.59, 3.37),
        (120, 35.30, 0, 35.30, 0, 0.18, -0.13, 0.18, -0.13),
    )
)

vdHG97_M92_metals = fp_zeros((21, 4))
for i in range(21):
    vdHG97_M92_metals[i, 0] = vdHG97_M92_metals_raw[i, 1] + \
        vdHG97_M92_metals_raw[i, 2]
    vdHG97_M92_metals[i, 1] = vdHG97_M92_metals_raw[i, 3] + \
        vdHG97_M92_metals_raw[i, 4]
    vdHG97_M92_metals[i, 2] = vdHG97_M92_metals_raw[i, 5] + \
        vdHG97_M92_metals_raw[i, 6]
    vdHG97_M92_metals[i, 3] = vdHG97_M92_metals_raw[i, 7] + \
        vdHG97_M92_metals_raw[i, 8]

vdHG97_M92_oxy = fp_zeros((21, 4))
for i in range(21):
    vdHG97_M92_oxy[i, 0] = vdHG97_M92_oxy_raw[i, 1] + vdHG97_M92_oxy_raw[i, 2]
    vdHG97_M92_oxy[i, 1] = vdHG97_M92_oxy_raw[i, 3] + vdHG97_M92_oxy_raw[i, 4]
    vdHG97_M92_oxy[i, 2] = vdHG97_M92_oxy_raw[i, 5] + vdHG97_M92_oxy_raw[i, 6]
    vdHG97_M92_oxy[i, 3] = vdHG97_M92_oxy_raw[i, 7] + vdHG97_M92_oxy_raw[i, 8]


# metal yield table accoring to INSERT
# TODO: instead of doing all that math in here, just put the output in here raw
vdHG97_M92_yields = fp_zeros((21, 9))
vdHG97_M92_yields[:, 0] = vdHG97_M92_metals_raw[:, 0]
vdHG97_M92_yields[:, 1::2] = vdHG97_M92_metals
vdHG97_M92_yields[:, 2::2] = vdHG97_M92_oxy


# metallicity cutoffs for the previous yield table
# Z < 0.0025 means use the first set, z < 0.006 use the second, etc.
vdHG97_M92_cutoffs = np.array((0.0025, 0.006, 0.01, np.inf))
