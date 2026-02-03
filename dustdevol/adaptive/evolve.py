from collections.abc import Callable
import numpy as np
from dustdevol.adaptive.generic import fp, fp_zeros, z_at_t
from scipy.interpolate import CubicHermiteSpline, CubicSpline


def evolve_2o(
    time_start,
    time_end,
    sfr_model,
    imf,
    inflow_model,
    outflow_model,
    recycling_model,
    grain_growth_model,
    destruction_model,
    ejecta_model,
    init_gas,
    init_star,
    init_metal,
    init_dust,
    model_params,
    absolute_tolerance,
    relative_tolerance,
):

    # guess at needed time step, and allocate space assuming that's
    # the time step.
    # TODO: More sophisticated first step choice
    dt = 0.001
    steps_guess = int(np.ceil((time_end - time_start) / dt))

    times = fp_zeros(steps_guess)
    times[:] = np.inf
    gas_masses = fp_zeros((steps_guess, len(init_gas)))
    star_masses = fp_zeros((steps_guess, len(init_star)))
    metal_masses = fp_zeros((steps_guess, len(init_metal)))
    dust_masses = fp_zeros((steps_guess, len(init_dust)))

    times[0] = time_start
    gas_masses[0] = init_gas
    star_masses[0] = init_star
    metal_masses[0] = init_metal
    dust_masses[0] = init_dust

    dgas_masses = fp_zeros((steps_guess, len(init_gas)))
    dstar_masses = fp_zeros((steps_guess, len(init_star)))
    dmetal_masses = fp_zeros((steps_guess, len(init_metal)))
    ddust_masses = fp_zeros((steps_guess, len(init_dust)))

    star_formation_rates = fp_zeros(steps_guess)

    # initialize the variables to actually be evolved during the sim loop
    t = time_start
    mgas = np.array(init_gas, dtype=fp)
    mstar = np.array(init_star, dtype=fp)
    mmetal = np.array(init_metal, dtype=fp)
    mdust = np.array(init_dust, dtype=fp)

    # initialize the variables to hold the change in each population due to
    # different phenomena, namely star formation, inflows, outflows,
    # outflow recycling, ejecta from dying (or perhaps still living) stars,
    # as well as grain growth and destruction in the ISM and clouds
    dmgas = fp_zeros(len(init_gas))
    dmstars = fp_zeros(len(init_star))
    dmmetal = fp_zeros(len(init_metal))
    dmdust = fp_zeros(len(init_dust))

    dmgas_astration = fp_zeros(len(init_gas))
    dmmetal_astration = fp_zeros(len(init_metal))
    dmdust_astration = fp_zeros(len(init_dust))

    dmgas_inflows = fp_zeros(len(init_gas))
    dmmetal_inflows = fp_zeros(len(init_metal))
    dmdust_inflows = fp_zeros(len(init_dust))

    dmgas_outflows = fp_zeros(len(init_gas))
    dmmetal_outflows = fp_zeros(len(init_metal))
    dmdust_outflows = fp_zeros(len(init_dust))

    dmgas_recycling = fp_zeros((len(init_gas)))
    dmmetal_recycling = fp_zeros((len(init_metal)))
    dmdust_recycling = fp_zeros((len(init_dust)))

    dmgas_ejecta = fp_zeros(len(init_gas))
    dmmetal_ejecta = fp_zeros(len(init_metal))
    dmdust_ejecta = fp_zeros(len(init_dust))

    dmdust_grain_growth = fp_zeros(len(init_dust))
    dmdust_destruction = fp_zeros(len(init_dust))

    i = 0
    cache = {}

    def gas_hist(t):
        return init_gas

    def star_hist(t):
        return init_star

    def metal_hist(t):
        return init_metal

    def dust_hist(t):
        return init_dust

    redshift = z_at_t(t)

    sfr = sfr_model(
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
    )

    star_formation_rates[0] = sfr

    def sfr_hist(t):
        return sfr

    cache["attempted_steps"] = 0

    while t < time_end:

        dmgas_astration = sfr * (mgas / mgas[0])
        dmmetal_astration = sfr * (mmetal / mgas[0])
        dmdust_astration = sfr * (mdust / mgas[0])

        dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
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
        )

        dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
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
        )

        dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
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
        )

        dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
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
        )

        dmdust_grain_growth = grain_growth_model(
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
        )

        dmdust_destruction = destruction_model(
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
        )

        dmgas = (
            -dmgas_astration
            + dmgas_inflows
            - dmgas_outflows
            + dmgas_recycling
            + dmgas_ejecta
        )
        dmstars = sfr - dmgas_ejecta
        dmmetal = (
            -dmmetal_astration
            + dmmetal_inflows
            - dmmetal_outflows
            + dmmetal_recycling
            + dmmetal_ejecta
        )
        dmdust = (
            -dmdust_astration
            + dmdust_inflows
            - dmdust_outflows
            + dmdust_recycling
            + dmdust_ejecta
            + dmdust_grain_growth
            - dmdust_destruction
        )

        dgas_masses[i] = dmgas
        dstar_masses[i] = dmstars
        dmetal_masses[i] = dmmetal
        ddust_masses[i] = dmdust

        while True:

            mgas_int = mgas + dmgas * dt
            mstar_int = mstar + dmstars * dt
            mmetal_int = mmetal + dmmetal * dt
            mdust_int = mdust + dmdust * dt

            gas_hist = CubicHermiteSpline(
                np.append(times[: i + 1], t + dt),
                np.vstack((gas_masses[: i + 1], mgas_int)),
                np.vstack((dgas_masses[: i + 1], dgas_masses[i])),
            )
            star_hist = CubicHermiteSpline(
                np.append(times[: i + 1], t + dt),
                np.vstack((star_masses[: i + 1], mstar_int)),
                np.vstack((dstar_masses[: i + 1], dstar_masses[i])),
            )
            metal_hist = CubicHermiteSpline(
                np.append(times[: i + 1], t + dt),
                np.vstack((metal_masses[: i + 1], mmetal_int)),
                np.vstack((dmetal_masses[: i + 1], dmetal_masses[i])),
            )
            dust_hist = CubicHermiteSpline(
                np.append(times[: i + 1], t + dt),
                np.vstack((dust_masses[: i + 1], mdust_int)),
                np.vstack((ddust_masses[: i + 1], ddust_masses[i])),
            )

            redshift = z_at_t(t + dt)

            sfr_int = sfr_model(
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
            )

            sfr_hist = CubicSpline(
                np.append(times[: i + 1], t + dt),
                np.append(star_formation_rates[: i + 1], sfr_int),
            )

            dmgas_astration = sfr_int * (mgas_int / mgas_int[0])
            dmmetal_astration = sfr_int * (mmetal_int / mgas_int[0])
            dmdust_astration = sfr_int * (mdust_int / mgas_int[0])

            dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
                model_params,
                sfr_int,
                imf,
                t + dt,
                redshift,
                mgas_int,
                mstar_int,
                mmetal_int,
                mdust_int,
                gas_hist,
                star_hist,
                metal_hist,
                dust_hist,
                sfr_hist,
                cache,
            )

            dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
                model_params,
                sfr_int,
                imf,
                t + dt,
                redshift,
                mgas_int,
                mstar_int,
                mmetal_int,
                mdust_int,
                gas_hist,
                star_hist,
                metal_hist,
                dust_hist,
                sfr_hist,
                cache,
            )

            dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
                model_params,
                sfr_int,
                imf,
                t + dt,
                redshift,
                mgas_int,
                mstar_int,
                mmetal_int,
                mdust_int,
                gas_hist,
                star_hist,
                metal_hist,
                dust_hist,
                sfr_hist,
                cache,
            )

            dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
                model_params,
                sfr_int,
                imf,
                t + dt,
                redshift,
                mgas_int,
                mstar_int,
                mmetal_int,
                mdust_int,
                gas_hist,
                star_hist,
                metal_hist,
                dust_hist,
                sfr_hist,
                cache,
            )

            dmdust_grain_growth = grain_growth_model(
                model_params,
                sfr_int,
                imf,
                t + dt,
                redshift,
                mgas_int,
                mstar_int,
                mmetal_int,
                mdust_int,
                gas_hist,
                star_hist,
                metal_hist,
                dust_hist,
                sfr_hist,
                cache,
            )

            dmdust_destruction = destruction_model(
                model_params,
                sfr_int,
                imf,
                t + dt,
                redshift,
                mgas_int,
                mstar_int,
                mmetal_int,
                mdust_int,
                gas_hist,
                star_hist,
                metal_hist,
                dust_hist,
                sfr_hist,
                cache,
            )

            dmgas_int = (
                -dmgas_astration
                + dmgas_inflows
                - dmgas_outflows
                + dmgas_recycling
                + dmgas_ejecta
            )
            dmstars_int = sfr_int - dmgas_ejecta
            dmmetal_int = (
                -dmmetal_astration
                + dmmetal_inflows
                - dmmetal_outflows
                + dmmetal_recycling
                + dmmetal_ejecta
            )
            dmdust_int = (
                -dmdust_astration
                + dmdust_inflows
                - dmdust_outflows
                + dmdust_recycling
                + dmdust_ejecta
                + dmdust_grain_growth
                - dmdust_destruction
            )

            mgas_fin = mgas + dt * (dmgas + dmgas_int) / 2
            mstar_fin = mstar + dt * (dmstars + dmstars_int) / 2
            mmetal_fin = mmetal + dt * (dmmetal + dmmetal_int) / 2
            mdust_fin = mdust + dt * (dmdust + dmdust_int) / 2

            err_gas = abs(dt * (-dmgas + dmgas_int) / 2)
            err_star = abs(dt * (-dmstars + dmstars_int) / 2)
            err_metal = abs(dt * (-dmmetal + dmmetal_int) / 2)
            err_dust = abs(dt * (-dmdust + dmdust_int) / 2)

            err = np.hstack((err_gas, err_star, err_metal, err_dust)) / (
                absolute_tolerance
                + relative_tolerance
                * np.maximum(
                    np.hstack((mgas_fin, mstar_fin, mmetal_fin, mdust_fin)),
                    np.hstack((mgas, mstar, mmetal, mdust)),
                )
            )

            err = np.sqrt((err**2).mean())

            times[i + 1] = t + dt

            cache["attempted_steps"] += 1

            dt = dt * max(0.5, min(2.0, 0.9 * (np.sqrt(1 / err))))

            if times[i + 1] - times[i] <= 1.1102230246251565e-14:
                raise RuntimeError(
                    "Stepsize too small for 64-bit precision. Either increasing or decreasing tolerance can help, though increasing is more likely."
                )

            if err <= 1:
                break

        if i + 2 >= len(times):

            n = len(times)
            times = np.append(times, np.full(n, np.inf))
            gas_masses = np.vstack((gas_masses, fp_zeros((n, len(init_gas)))))
            star_masses = np.vstack((star_masses, fp_zeros((n, len(init_star)))))
            metal_masses = np.vstack((metal_masses, fp_zeros((n, len(init_metal)))))
            dust_masses = np.vstack((dust_masses, fp_zeros((n, len(init_dust)))))

            dgas_masses = np.vstack((dgas_masses, fp_zeros((n, len(init_gas)))))
            dstar_masses = np.vstack((dstar_masses, fp_zeros((n, len(init_star)))))
            dmetal_masses = np.vstack((dmetal_masses, fp_zeros((n, len(init_metal)))))
            ddust_masses = np.vstack((ddust_masses, fp_zeros((n, len(init_dust)))))

            star_formation_rates = np.append(star_formation_rates, fp_zeros(n))

        t = times[i + 1]
        mgas = mgas_fin
        mstar = mstar_fin
        mmetal = mmetal_fin
        mdust = mdust_fin

        gas_masses[i + 1] = mgas
        star_masses[i + 1] = mstar
        metal_masses[i + 1] = mmetal
        dust_masses[i + 1] = mdust

        i += 1

        gas_hist = CubicHermiteSpline(
            times[: i + 1],
            gas_masses[: i + 1],
            np.vstack((dgas_masses[:i], dgas_masses[i - 1])),
        )
        star_hist = CubicHermiteSpline(
            times[: i + 1],
            star_masses[: i + 1],
            np.vstack((dstar_masses[:i], dstar_masses[i - 1])),
        )
        metal_hist = CubicHermiteSpline(
            times[: i + 1],
            metal_masses[: i + 1],
            np.vstack((dmetal_masses[:i], dmetal_masses[i - 1])),
        )
        dust_hist = CubicHermiteSpline(
            times[: i + 1],
            dust_masses[: i + 1],
            np.vstack((ddust_masses[:i], ddust_masses[i - 1])),
        )

        sfr = sfr_model(
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
        )

        star_formation_rates[i] = sfr

        sfr_hist = CubicSpline(
            times[: i + 1],
            star_formation_rates[: i + 1],
        )

    dmgas_astration = sfr * (mgas / mgas[0])
    dmmetal_astration = sfr * (mmetal / mgas[0])
    dmdust_astration = sfr * (mdust / mgas[0])

    dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
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
    )

    dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
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
    )

    dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
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
    )

    dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
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
    )

    dmdust_grain_growth = grain_growth_model(
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
    )

    dmdust_destruction = destruction_model(
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
    )

    dmgas = (
        -dmgas_astration
        + dmgas_inflows
        - dmgas_outflows
        + dmgas_recycling
        + dmgas_ejecta
    )
    dmstars = sfr - dmgas_ejecta
    dmmetal = (
        -dmmetal_astration
        + dmmetal_inflows
        - dmmetal_outflows
        + dmmetal_recycling
        + dmmetal_ejecta
    )
    dmdust = (
        -dmdust_astration
        + dmdust_inflows
        - dmdust_outflows
        + dmdust_recycling
        + dmdust_ejecta
        + dmdust_grain_growth
        - dmdust_destruction
    )

    dgas_masses[i] = dmgas
    dstar_masses[i] = dmstars
    dmetal_masses[i] = dmmetal
    ddust_masses[i] = dmdust

    to_keep = times != np.inf

    results = {
        "times": times[to_keep],
        "gas_masses": gas_masses[to_keep],
        "star_masses": star_masses[to_keep],
        "metal_masses": metal_masses[to_keep],
        "dust_masses": dust_masses[to_keep],
        "dgas_masses": dgas_masses[to_keep],
        "dstar_masses": dstar_masses[to_keep],
        "dmetal_masses": dmetal_masses[to_keep],
        "ddust_masses": ddust_masses[to_keep],
        "sfr": star_formation_rates[to_keep],
        "cache": cache,
    }

    return results


"""
def evolve_sfe(
    time_start: fp,
    time_end: fp,
    times: Callable[[dict], list[fp]],
    sfe_model: Callable[[dict], list[fp]],
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
"""

    # trim times array to exclude points before the start or after the end
    times = np.hstack(((fp(0)), times[(times < time_end) & (times > time_start)]))

    # calculate sfes, and reserve space for the sfrs
    sfe = sfe_model(model_params, times)
    sfr = fp_zeros(len(sfe))

    # reserve space for the results, with a row for each timestep (including 0)
    # and a column for the times, as well as each subcategory of gas, metal,
    # stars, and dust, and put in what you already have (times and init values)
    results = fp_zeros(
        (
            len(times),
            len(init_gas) + len(init_star) +
            len(init_metal) + len(init_dust) + 1,
        ),
    )
    results[:, 0] = times
    results[0, 1:] = np.hstack((init_gas, init_star, init_metal, init_dust))

    # initialize the variables to actually be evolved during the sim loop
    mgas = np.array(init_gas, dtype=fp)
    mstar = np.array(init_star, dtype=fp)
    mmetal = np.array(init_metal, dtype=fp)
    mdust = np.array(init_dust, dtype=fp)

    # initialize the variables to hold the change in each population due to
    # different phenomena, namely star formation, inflows, outflows,
    # outflow recycling, ejecta from dying (or perhaps still living) stars,
    # as well as grain growth and destruction in the ISM and clouds
    dmstars = fp_zeros(len(init_star))
    dmgas_astration = fp_zeros(len(init_gas))
    dmmetal_astration = fp_zeros(len(init_metal))
    dmdust_astration = fp_zeros(len(init_dust))

    dmgas_inflows = fp_zeros(len(init_gas))
    dmmetal_inflows = fp_zeros(len(init_metal))
    dmdust_inflows = fp_zeros(len(init_dust))

    dmgas_outflows = fp_zeros(len(init_gas))
    dmmetal_outflows = fp_zeros(len(init_metal))
    dmdust_outflows = fp_zeros(len(init_dust))

    dmgas_recycling = fp_zeros((len(init_gas)))
    dmmetal_recycling = fp_zeros((len(init_metal)))
    dmdust_recycling = fp_zeros((len(init_dust)))

    dmgas_ejecta = fp_zeros(len(init_gas))
    dmmetal_ejecta = fp_zeros(len(init_metal))
    dmdust_ejecta = fp_zeros(len(init_dust))

    dmdust_grain_growth = fp_zeros(len(init_dust))
    dmdust_destruction = fp_zeros(len(init_dust))

    # initialize time and begin the loop

    for i in range(0, len(times) - 1):

        # find size of current time step, redshift, and
        # calculate sfr from sfe and current parameters
        t = times[i]
        next_t = times[i + 1]
        dt = next_t - t

        redshift = z_at_t(t)

        sfr[i] = (
            sfe[i]
            * mgas
            * (max(1e5, mstar) / 1e9) ** 0.25
            * (1 + np.exp(mstar / (10 * mgas))) ** -3
            * (1 + redshift) ** -1
        )

        # change in pops due to star formation
        dmgas_astration = sfr[i] * (mgas / mgas[0])
        dmmetal_astration = sfr[i] * (mmetal / mgas[0])
        dmdust_astration = sfr[i] * (mdust / mgas[0])

        # change in pops due to inflows
        dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
            model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
        )

        # change in pops due to outflows
        dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
            model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
        )

        # change in pops due to outflow recycling
        dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
            model_params,
            times,
            i,
            redshift,
            dmgas_outflows,
            dmmetal_outflows,
            dmdust_outflows,
            mgas,
            mstar,
            mmetal,
            mdust,
        )

        # change in pops due to stellar ejecta
        dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
            model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
        )

        # change in pops due to grain growth
        dmdust_grain_growth = grain_growth_model(
            model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
        )

        # change in pops due to dust destruction
        dmdust_destruction = destruction_model(
            model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
        )

        # change in stellar mass due to stars forming and dying
        dmstars = sfr[i] - dmgas_ejecta

        # collate together the effects of everything
        mgas += (
            -dmgas_astration
            + dmgas_inflows
            - dmgas_outflows
            + dmgas_recycling
            + dmgas_ejecta
        ) * dt

        mstar += dmstars * dt

        mmetal += (
            -dmmetal_astration
            + dmmetal_inflows
            - dmmetal_outflows
            + dmmetal_recycling
            + dmmetal_ejecta
        ) * dt

        mdust += (
            -dmdust_astration
            + dmdust_inflows
            - dmdust_outflows
            + dmdust_recycling
            + dmdust_ejecta
            + dmdust_grain_growth
            - dmdust_destruction
        ) * dt

        # sanity checks:
        # is there a non-negligible amount of gas remaining?
        if mgas[0] / (mstar[0] + mgas[0]) < 0.0001:
            print("No gas remaining. Terminating early at %f." % t)
            return results

        # non-negative metals remaining?
        if mmetal[0] < 0:
            print("No metal remaining. Terminating early at %f" % t)
            return results

        # non-negative dust remaining?
        if mdust[0] < 0:
            print("No dust remaining. Terminating early at %f." % t)
            return results

        # if we are sane, store
        results[1 + i, 1:] = np.hstack((mgas, mstar, mmetal, mdust))

    return results


def evolve_sfr(
    time_start: fp,
    time_end: fp,
    times: Callable[[dict], list[fp]],
    sfr_model: Callable[[dict], list[fp]],
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
"""

    # trim times array to exclude points before the start or after the end
    # and also include t = 0
    times = np.hstack(((fp(0)), times[(times < time_end) & (times > time_start)]))

    # calculate sfes, and reserve space for the sfrs
    sfr = sfr_model(model_params, times)

    # reserve space for the results, with a row for each timestep (including 0)
    # and a column for the times, as well as each subcategory of gas, metal,
    # stars, and dust, and put in what you already have (times and init values)
    results = fp_zeros(
        (
            len(times),
            len(init_gas) + len(init_star) +
            len(init_metal) + len(init_dust) + 1,
        ),
    )
    results[:, 0] = times
    results[0, 1:] = np.hstack((init_gas, init_star, init_metal, init_dust))

    # initialize the variables to actually be evolved during the sim loop
    mgas = np.array(init_gas, dtype=fp)
    mstar = np.array(init_star, dtype=fp)
    mmetal = np.array(init_metal, dtype=fp)
    mdust = np.array(init_dust, dtype=fp)

    # initialize the variables to hold the change in each population due to
    # different phenomena, namely star formation, inflows, outflows,
    # outflow recycling, ejecta from dying (or perhaps still living) stars,
    # as well as grain growth and destruction in the ISM and clouds
    dmstars = fp_zeros(len(init_star))
    dmgas_astration = fp_zeros(len(init_gas))
    dmmetal_astration = fp_zeros(len(init_metal))
    dmdust_astration = fp_zeros(len(init_dust))

    dmgas_inflows = fp_zeros(len(init_gas))
    dmmetal_inflows = fp_zeros(len(init_metal))
    dmdust_inflows = fp_zeros(len(init_dust))

    dmgas_outflows = fp_zeros(len(init_gas))
    dmmetal_outflows = fp_zeros(len(init_metal))
    dmdust_outflows = fp_zeros(len(init_dust))

    dmgas_recycling = fp_zeros((len(init_gas)))
    dmmetal_recycling = fp_zeros((len(init_metal)))
    dmdust_recycling = fp_zeros((len(init_dust)))

    dmgas_ejecta = fp_zeros(len(init_gas))
    dmmetal_ejecta = fp_zeros(len(init_metal))
    dmdust_ejecta = fp_zeros(len(init_dust))

    dmdust_grain_growth = fp_zeros(len(init_dust))
    dmdust_destruction = fp_zeros(len(init_dust))

    # initialize time and begin the loop

    for i in range(0, len(times) - 1):

        # find size of current time step, redshift, and
        # calculate sfr from sfe and current parameters
        t = times[i]
        next_t = times[i + 1]
        dt = next_t - t

        redshift = z_at_t(t)

        # change in pops due to star formation
        dmgas_astration = sfr[i] * (mgas / mgas[0])
        dmmetal_astration = sfr[i] * (mmetal / mgas[0])
        dmdust_astration = sfr[i] * (mdust / mgas[0])

        # change in pops due to inflows
        dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
            model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
        )

        # change in pops due to outflows
        dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
            model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
        )

        # change in pops due to outflow recycling
        dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
            model_params,
            times,
            i,
            redshift,
            dmgas_outflows,
            dmmetal_outflows,
            dmdust_outflows,
            mgas,
            mstar,
            mmetal,
            mdust,
        )

        # change in pops due to stellar ejecta
        dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
            model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
        )

        # change in pops due to grain growth
        dmdust_grain_growth = grain_growth_model(
            model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
        )

        # change in pops due to dust destruction
        dmdust_destruction = destruction_model(
            model_params, sfr, imf, times, i, redshift, mgas, mstar, mmetal, mdust
        )

        # change in stellar mass due to stars forming and dying
        dmstars = sfr[i] - dmgas_ejecta

        # collate together the effects of everything
        mgas += (
            -dmgas_astration
            + dmgas_inflows
            - dmgas_outflows
            + dmgas_recycling
            + dmgas_ejecta
        ) * dt

        mstar += dmstars * dt

        mmetal += (
            -dmmetal_astration
            + dmmetal_inflows
            - dmmetal_outflows
            + dmmetal_recycling
            + dmmetal_ejecta
        ) * dt

        mdust += (
            -dmdust_astration
            + dmdust_inflows
            - dmdust_outflows
            + dmdust_recycling
            + dmdust_ejecta
            + dmdust_grain_growth
            - dmdust_destruction
        ) * dt

        # sanity checks:
        # is there a non-negligible amount of gas remaining?
        if mgas[0] / (mstar[0] + mgas[0]) < 0.0001:
            print("No gas remaining. Terminating early at %f." % t)
            return results

        # non-negative metals remaining?
        if mmetal[0] < 0:
            print("No metal remaining. Terminating early at %f" % t)
            return results

        # non-negative dust remaining?
        if mdust[0] < 0:
            print("No dust remaining. Terminating early at %f." % t)
            return results

        # if we are sane, store
        results[1 + i, 1:] = np.hstack((mgas, mstar, mmetal, mdust))

    return results
"""
