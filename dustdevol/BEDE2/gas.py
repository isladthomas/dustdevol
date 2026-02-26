from numpy import exp, where, clip, log10, linspace, minimum, maximum, column_stack
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import root
from dustdevol.generic import fp_zeros, z_at_t, fp, fp_array


def BEDE_inflow(
    model_params,
    sfr,
    imf,
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
    sfr_hist,
    cache,
):
    """
    calculate gas inflow according to De Vis 2020, allowing for enrichment
    if inflow metal and dust frac are not specified, assume 0

    Paramters
    ---------
    model_params : dict
        - \"total_inflow\": total mass to be accreted as inflows
        - \"infall_time\": timescale for infalls. In the limit that
        tot_infall_time goes to infty, the time it takes for (1 - 1/e) of the
        remaining inflow to accrete.
        - \"tot_infall_time\": total timescale for infalls. Specifically, the
        time it takes to accrete total_inflow.
        optional:
        - \"inflow_metal\": fraction of inflows in the form of metal
                            should have same shape as init_metal
        - \"inflow_dust\": fraction of inflows in the form of dust
                           should have same shape as init_dust

    Returns
    -------
    out : (g,), (m,), (d,)
          gas, metals, and dust gained from inflows in Msol/Gyr
    """

    gas_inflow = (
        model_params["total_inflow"]
        * exp(-t / model_params["infall_time"])
        / model_params["infall_time"]
        * (1 - exp(-model_params["tot_infall_time"] / model_params["infall_time"]))
    )

    # if metal and dust frac not specified, set to zero
    # (so the error stops happening and we can go on quicker)
    try:
        metal_inflow = gas_inflow * model_params["inflow_metal"]
    except KeyError:
        model_params["inflow_metal"] = fp_zeros(len(mmetal))
        metal_inflow = fp_zeros(len(mmetal))

    try:
        dust_inflow = gas_inflow * model_params["inflow_dust"]
    except KeyError:
        model_params["inflow_dust"] = fp_zeros(len(mdust))
        dust_inflow = fp_zeros(len(mdust))

    return gas_inflow, metal_inflow, dust_inflow


def Nelson_outflow(
    model_params,
    sfr,
    imf,
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
    sfr_hist,
    cache,
):
    """
    Calculate gas outflow according to De Vis 2020 using nelson's prescription
    assumes outflows are a multiple of sfr, but which varies with stellar mass
    and redshift. Calculates total outflow in velocity bins of 0-150, 150-300,
    and >300 km/s

    Parameters
    ----------
    model_params : dict
    optional:
        - \"outflow_metal\": multiplies metal outflow assuming outflows have
        same gas-metal ratio as rest of galaxy
        - \"outflow_dust\": same as above with dust instead of metal

    Returns
    -------
    out : (g,), (m,), (d,)
          Gas, metal, and dust lost (positive) to outflows
    """

    eta = mass_loading(fp_array([redshift]), mstar)
    gas_outflow = minimum((sfr * 10**eta).sum(), mgas * 0.5 / 0.03)

    # if metal and dust frac not specified, set to zero
    # (so the error stops happening and we can go on quicker)
    try:
        metal_outflow = (mmetal / mgas[0]) * \
            gas_outflow * model_params["outflow_metal"]
    except KeyError:
        model_params["outflow_metal"] = fp_zeros(len(mmetal))
        metal_outflow = fp_zeros(len(mmetal))

    try:
        dust_outflow = (mdust / mgas[0]) * \
            gas_outflow * model_params["outflow_dust"]
    except KeyError:
        model_params["outflow_dust"] = fp_zeros(len(mdust))
        dust_outflow = fp_zeros(len(mdust))

    return gas_outflow, metal_outflow, dust_outflow


def BEDE_recycling(
    model_params,
    sfr,
    imf,
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
    sfr_hist,
    cache,
):

    ejection_times = find_recycle_times(
        t, redshift, star_hist, model_params["recycling_scaling"], cache
    )

    return nelson_recyc(
        star_hist(t - ejection_times),
        z_at_t(t - ejection_times),
        where(t - ejection_times > 0, sfr_hist(t - ejection_times), 0),
        metal_hist(t - ejection_times)
        / gas_hist(t - ejection_times)[0]
        * model_params["outflow_metal"],
        dust_hist(t - ejection_times)
        / gas_hist(t - ejection_times)[0]
        * model_params["outflow_dust"],
        exp(
            -model_params["IGM_loss"]
            * model_params["recycling_scaling"]
            * ejection_times
        ),
    )


def find_recycle_times(t, redshift, star_hist, scaling, cache):

    try:
        tau0 = cache["IGM_lifetimes"]
    except KeyError:
        cache["IGM_lifetimes"] = IGM_lifetimes(
            fp_array([redshift]), fp(0)) * scaling
        tau0 = cache["IGM_lifetimes"]

    if abs(
        (
            IGM_lifetimes(
                z_at_t(t - tau0),
                star_hist(t - tau0)[:, 0],
            )
            * scaling
            - tau0
        )
    ).max() > fp(1e-3):
        soln = root(
            lambda tau: IGM_lifetimes(
                z_at_t(t - tau),
                star_hist(t - tau)[:, 0],
            )
            * scaling
            - tau,
            tau0,
        ).x
    else:
        soln = tau0

    cache["IGM_lifetimes"] = soln

    return soln


def nelson_recyc(mstar, redshift, sfr, metal_frac, dust_frac, escape_factor):

    eta = mass_loading(redshift, mstar[:, 0])
    gas_outflow = escape_factor * sfr * 10**eta

    metal_outflow = gas_outflow[:, None] * metal_frac
    dust_outflow = gas_outflow[:, None] * dust_frac

    return gas_outflow.sum(axis=0), metal_outflow.sum(axis=0), dust_outflow.sum(axis=0)


Nelson_Z = fp_array((0, 0.5, 1.0, 2.0, 4.0))
Nelson_logMstar = linspace(7.5, 11.5, 20, dtype=fp)

Nelson_logEta_cubic = fp_array(
    (
        (
            [1.546, -0.555, -3.0],
            [1.47581629, -0.16026175, -3.00018479],
            [1.40792215, 0.06963564, -2.99185357],
            [1.3631994, 0.21262851, -3.03072357],
            [1.334084, 0.29154466, -2.87952159],
            [1.29054831, 0.36685188, -0.01730405],
            [1.2380357, 0.39968844, -0.16451073],
            [1.19349856, 0.41609687, -0.03717971],
            [1.15587404, 0.39798904, -0.0348975],
            [1.11464938, 0.39479631, -0.0148968],
            [1.06553257, 0.34948257, -0.02648604],
            [1.03030602, 0.28479258, -0.06568598],
            [1.04163401, 0.22513755, -0.11056786],
            [1.06883298, 0.22294097, -0.17509951],
            [1.14400653, 0.38419919, -0.11607334],
            [1.38925289, 0.79451548, 0.20711497],
            [1.78078563, 1.32128575, 0.8013349],
            [2.12509235, 1.7787355, 1.39521984],
            [2.25612429, 2.02772262, 1.79907872],
            [2.31365232, 2.15135481, 1.94957012],
        ),
        (
            [1.67000000e00, 1.94000000e-01, -3.00000000e00],
            [1.60584451e00, 3.44387366e-01, -2.99931757e00],
            [1.54598151e00, 4.86402442e-01, -3.03009812e00],
            [1.49161660e00, 5.77462123e-01, -2.98528315e00],
            [1.44045299e00, 6.19325290e-01, -3.53314210e-04],
            [1.38495020e00, 6.47130198e-01, 4.79505278e-02],
            [1.33200517e00, 6.55017698e-01, 1.15266119e-01],
            [1.25271431e00, 6.48976280e-01, 1.33926664e-01],
            [1.17345680e00, 6.24039042e-01, 1.51254514e-01],
            [1.09543228e00, 5.90105735e-01, 1.82722039e-01],
            [1.03736005e00, 5.36359156e-01, 1.92897805e-01],
            [9.71283498e-01, 4.64637644e-01, 1.71066431e-01],
            [9.48136637e-01, 4.12389496e-01, 1.36135417e-01],
            [9.51459989e-01, 4.12489072e-01, 1.42192630e-01],
            [9.98698571e-01, 4.55274524e-01, 1.43705148e-01],
            [1.08418517e00, 5.98994872e-01, 2.36147851e-01],
            [1.21912147e00, 8.11900333e-01, 4.31993100e-01],
            [1.49021337e00, 1.21456573e00, 9.12921775e-01],
            [1.65435940e00, 1.45952883e00, 1.25274782e00],
            [1.85558880e00, 1.73769622e00, 1.60345314e00],
        ),
        (
            [1.7001882, 0.32308949, -3.0966832],
            [1.63483973, 0.52739414, -3.00242343],
            [1.57067568, 0.64237352, -2.98999008],
            [1.50449413, 0.6892236, -0.03188168],
            [1.43940725, 0.71525216, 0.06425768],
            [1.38222305, 0.74132154, 0.13913223],
            [1.30874407, 0.75801212, 0.16784106],
            [1.21730331, 0.74257324, 0.19687024],
            [1.12117849, 0.71221263, 0.24694528],
            [1.0343855, 0.66226319, 0.28922654],
            [0.94770569, 0.59760653, 0.31686552],
            [0.88701074, 0.52533523, 0.29871211],
            [0.84167853, 0.46693355, 0.25395209],
            [0.81293337, 0.41927057, 0.19635689],
            [0.81857065, 0.40574628, 0.1601138],
            [0.87440441, 0.46894574, 0.18964212],
            [0.94425954, 0.61153189, 0.30782188],
            [1.07942309, 0.83427884, 0.55920742],
            [1.36356076, 1.22433711, 1.05895902],
            [1.58136746, 1.51891897, 1.42604006],
        ),
        (
            [1.678, 0.756, -0.525],
            [1.60044146, 0.82850126, -0.26105065],
            [1.52873833, 0.87256543, -0.0688602],
            [1.46685953, 0.8954766, 0.07600825],
            [1.39550803, 0.89031289, 0.19974768],
            [1.33451778, 0.88360316, 0.24402341],
            [1.26002168, 0.86461291, 0.26182216],
            [1.18025582, 0.84305981, 0.31446705],
            [1.07552779, 0.79142003, 0.39538201],
            [0.96302805, 0.73120788, 0.45664922],
            [0.85788391, 0.65186975, 0.44061466],
            [0.76128914, 0.57099534, 0.39880341],
            [0.67877346, 0.48804481, 0.33018322],
            [0.59101583, 0.41093587, 0.26408193],
            [0.54405833, 0.34580819, 0.19089191],
            [0.48510121, 0.26516464, 0.08990371],
            [0.58079959, 0.37375047, 0.18177004],
            [0.79348809, 0.63616125, 0.45380481],
            [1.00858561, 0.91780821, 0.7760572],
            [1.22902535, 1.18806754, 1.1019236],
        ),
        (
            [1.42715385, 0.97324555, 0.20304209],
            [1.39890903, 0.98600551, 0.30821234],
            [1.35777934, 0.98195953, 0.37423985],
            [1.31367939, 0.97190993, 0.40817613],
            [1.27162945, 0.95852908, 0.42106145],
            [1.22567009, 0.94448547, 0.41129257],
            [1.17394374, 0.92956978, 0.41811231],
            [1.10570896, 0.90541119, 0.46087909],
            [1.01932078, 0.85455322, 0.50387584],
            [0.89863074, 0.7727499, 0.52067268],
            [0.80373171, 0.70669426, 0.51858014],
            [0.64800284, 0.57560771, 0.43580818],
            [0.5304823, 0.46803745, 0.35614491],
            [0.4123874, 0.33693785, 0.2309301],
            [0.29645622, 0.19905993, 0.08424364],
            [0.17864634, 0.07115304, -0.05029437],
            [0.02011251, -0.06703873, -0.17236328],
            [-0.2181701, -0.23596273, -0.28181535],
            [-0.57522629, -0.45606629, -0.37850281],
            [-1.09008089, -0.74779674, -0.46227793],
        ),
    )
)

Nelson_interp_cubic_extrap = RegularGridInterpolator(
    (Nelson_Z, Nelson_logMstar), Nelson_logEta_cubic, method="cubic"
)


def Nelson_interp_cubic(redshift, mstar):
    z = clip(redshift, 0, 4.0)
    m = clip(log10(mstar), 7.5, 11.5)
    return Nelson_interp_cubic_extrap(column_stack((z, m)))[0, :]


mass_loading = Nelson_interp_cubic

TNG100_Z = fp_array([0.2, 0.5, 1.0, 2.0, 4.0])
TNG100_logMstar = fp_array([8.0, 9.0, 9.5, 10.0, 10.5, 11.0, 11.5])

TNG100_lifetimes = fp_array(
    [
        [
            [0.6658, 20.0, 20.0],
            [0.3979, 4.28, 20.0],
            [0.2553, 2.902, 20.0],
            [0.1584, 1.4929, 20.0],
            [0.1031, 0.7345, 4.04],
            [0.0581, 0.1512, 3.344],
            [0.026, 0.0361, 0.189],
        ],
        [
            [0.7921, 20.0, 20.0],
            [0.4624, 5.336, 20.0],
            [0.2704, 3.552, 20.0],
            [0.1752, 1.8875, 20.0],
            [0.11, 0.8809, 4.804],
            [0.0567, 0.1418, 3.698],
            [0.0261, 0.0373, 0.17],
        ],
        [
            [1.0006, 20.0, 20.0],
            [0.5439, 5.999, 20.0],
            [0.3552, 4.36, 20.0],
            [0.2109, 2.36, 20.0],
            [0.1199, 1.3105, 10.378],
            [0.0592, 0.1441, 4.84],
            [0.0242, 0.0318, 0.11],
        ],
        [
            [1.3402, 20.0, 20.0],
            [0.6306, 6.996, 20.0],
            [0.4444, 5.694, 20.0],
            [0.2629, 3.0258, 20.0],
            [0.1524, 1.5289, 10.894],
            [0.0789, 0.2254, 5.978],
            [0.0353, 0.0538, 0.839],
        ],
        [
            [1.9748, 20.0, 20.0],
            [0.9039, 7.97, 20.0],
            [0.7958, 6.978, 20.0],
            [0.4588, 4.47, 20.0],
            [0.206, 2.4269, 20.0],
            [0.1066, 0.4666, 4.286],
            [0.0353, 0.0538, 0.839],
        ],
    ]
)


IGM_lifetimes_extrap = RegularGridInterpolator(
    (TNG100_Z, TNG100_logMstar), TNG100_lifetimes, method="cubic"
)


def IGM_lifetimes(redshift, mstar):
    z = clip(redshift, 0.2, 4.0)
    m = clip(log10(mstar), 8.0, 11.5)
    return maximum(IGM_lifetimes_extrap(column_stack((z, m)))[0, :], 0)
