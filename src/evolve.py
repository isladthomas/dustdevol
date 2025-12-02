from collections.abc import Callable
import numpy as np

# define working precision
fp = np.float64


def fp_zeros(shape):
    return np.zeros(shape, dtype=fp)


def evolve_sfe(
    time_start: fp,
    time_end: fp,
    times: Callable[[dict], list[fp]],
    sfe_model: Callable[[dict], list[fp]],
    sn_model: Callable[[dict], list[fp]],
    imf: Callable[[dict], list[fp]],
    inflow_model: Callable[[dict], list[fp]],
    outflow_model: Callable[[dict, ...], list[fp]],
    recycling_model: Callable[[dict, ...], list[fp]],
    grain_growth_model: Callable[[dict, ...], list[fp]],
    destruction_model: Callable[[dict, ...], list[fp]],
    ejecta_model: Callable[[dict, ...], list[fp]],
    init_gas: list[fp],
    init_star: list[fp],
    init_metal: list[fp],
    init_dust: list[fp],
    # integrator: Callable[[...], list[fp]],
    model_params: dict,
):
    """
    The main evolution loop for the galaxy model described by the various
    functions passed to it, as well as the contents of "modelParams."

    - time_start: start time for the simulation
    - time_end: end time for the simulation
    - times: function which takes in the dictionary of model params
             and outputs an array of timesteps which the final
             integrator will use
             Plan to replace with simply a grid of points that
             will be populated by sfr, inflows, etc. and then interpolated
             in an adaptive timestep scheme
    - sfr_model: function which takes in the dictionary of model params
           and outputs the sfr at each timestep in "times"
    - inflow_model: likewise but for inflows
    - outflow_model: function which takes in model params, sfr, current time,
                     and current mass of each population and outputs outflow
                     of each population
    - recycling_model: function which takes in model params, current time,
                       outflows, and current mass of each population and
                       populates the "recycling" array at each timestep in
                       "times" using the outflows from this timestep
    - grain_growth_model: function which takes in model params, sfr,
                          and current mass of each population and
                          outputs the rate of grain growth in each dust
                          population
    - destruction_model: function which takes in model params, sfr,
                         and current mass of each population and
                         outputs the rate of photofragmentation in each
                         dust population
    - ejecta_model: takes in model params, current time, sfr, imf,
                    and current mass of each population and outputs
                    mass of each population from ejecta from dying stars
    - init_gas: variable length array denoting the initial mass of
                various gas populations.
                First element should be total gas
    - init_star: same as "init_gas" but for stellar populations
    - init_metal: likewise but for metals
    - init_dust: likewise but for dust
    - model_params: dictionary containing all the various parameters for each
                    of the functions passed to evolve
    """

    times = times[(times < time_end) & (times > time_start)]

    sfe = sfe_model(model_params, times)
    sfr = fp_zeros(sfe)
    inflows = inflow_model(model_params, times)

    results = fp_zeros(
        (
            len(times) + 1,
            len(init_gas) + len(init_star) + len(init_metal) + len(init_dust) + 1,
        ),
    )
    results[:, 0] = times
    results[0, 1:] = np.hstack(init_gas, init_star, init_metal, init_dust)

    mgas = init_gas
    mstar = init_star
    mmetal = init_metal
    mdust = init_dust

    dmstars = fp_zeros(init_star)

    dmgas_astration = fp_zeros(init_gas)
    dmmetal_astration = fp_zeros(init_metal)
    dmdust_astration = fp_zeros(init_dust)

    dmgas_inflows = fp_zeros(init_gas)
    dmmetal_inflows = fp_zeros(init_metal)
    dmdust_inflows = fp_zeros(init_dust)

    dmgas_outflows = fp_zeros(init_gas)
    dmmetal_outflows = fp_zeros(init_metal)
    dmdust_outflows = fp_zeros(init_dust)

    dmgas_recycling = fp_zeros((len(times), init_gas))
    dmmetal_recycling = fp_zeros((len(times), init_metal))
    dmdust_recycling = fp_zeros((len(times), init_dust))

    dmgas_ejecta = fp_zeros(init_gas)
    dmmetal_ejecta = fp_zeros(init_metal)
    dmdust_ejecta = fp_zeros(init_dust)

    dmdust_grain_growth = fp_zeros(init_dust)
    dmdust_destruction = fp_zeros(init_dust)

    prev_t = 0

    for i in range(0, len(times)):

        t = times[i]
        dt = t - prev_t

        redshift = z_at_t(t)

        sfr[i] = (
            sfe[i]
            * mgas
            * (max(1e5, mstar) / 1e9) ** 0.25
            * (1 + np.exp(mstar / (10 * mgas))) ** -3
            * (1 + redshift) ** -1
        )

        dmgas_astration = sfr[i] * (mgas / mgas[0])
        dmmetal_astration = sfr[i] * (mmetal / mgas[0])
        dmdust_astration = sfr[i] * (mdust / mgas[0])

        dmgas_inflows = inflows[i] * model_params["inflow_gas_frac"]
        dmmetal_inflows = inflows[i] * model_params["inflow_metal_frac"]
        dmdust_inflows = inflows[i] * model_params["inflow_dust_frac"]

        dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
            model_params, sfr, times, i, redshift, mgas, mstar, mmetal, mdust
        )

        dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
            model_params,
            times,
            i,
            redshift,
            dmgas_recycling,
            dmmetal_recycling,
            dmdust_recycling,
            dmgas_outflows,
            dmmetal_outflows,
            dmdust_outflows,
            mgas,
            mstar,
            mmetal,
            mdust,
        )

        dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
            model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
        )

        dmdust_grain_growth = grain_growth_model(
            model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
        )

        dmdust_destruction = destruction_model(
            model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
        )

        dmstars = sfr[i] - dmgas_ejecta

        mgas += (
            -dmgas_astration
            + dmgas_inflows
            - dmgas_outflows
            + dmgas_recycling[i]
            + dmgas_ejecta
        ) * dt

        mstar += dmstars * dt

        mmetal += (
            -dmmetal_astration
            + dmmetal_inflows
            - dmmetal_outflows
            + dmmetal_recycling[i]
            + dmmetal_ejecta
        ) * dt

        mdust += (
            -dmdust_astration
            + dmdust_inflows
            - dmdust_outflows
            + dmdust_recycling[i]
            + dmdust_ejecta
            + dmdust_grain_growth
            - dmdust_destruction
        ) * dt

        if mgas[0] / (mstar[0] + mgas[0]) < 0.0001:
            print("No gas remaining. Terminating early at %f." % t)
            return results

        if mmetal[0] < 0:
            print("No metal remaining. Terminating early at %f" % t)
            return results

        if mdust[0] < 0:
            print("No dust remaining. Terminating early at %f." % t)
            return results

        results[1 + i, 1:] = np.hstack(init_gas, init_star, init_metal, init_dust)

    return results
