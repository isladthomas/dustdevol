import numpy as np
from dustdevol.adaptive.generic import fp, fp_zeros, z_at_t
from scipy.interpolate import CubicHermiteSpline, CubicSpline
import sys


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
    dt = 0.003
    steps_guess = int(np.ceil((time_end - time_start) / dt))

    # Create arrays to store all outputs
    # The derivatives are needed to create hermite interpolants for the output
    # so we can get a good guess as to outputs even where our sim doesn't visit.
    # Hermite int. has error order O(h^4), while our method only has O(h^2),
    # so not really any extra error introduced by interpolating.
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

    # Higher order methods have "mini-steps," so init arrays to store those
    mgas_int = np.array(init_gas, dtype=fp)
    mstar_int = np.array(init_star, dtype=fp)
    mmetal_int = np.array(init_metal, dtype=fp)
    mdust_int = np.array(init_dust, dtype=fp)

    # package each step for ease of reading in below function calls
    # takes it from 17 lines per call to 3
    y1 = [mgas, mstar, mmetal, mdust]
    y2 = [mgas_int, mstar_int, mmetal_int, mdust_int]

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

    # Keep track of what step we're on, and also provide
    # a dictionary that lets functions give additional output/
    # store certain results in memory to speed up later computation.
    i = 0
    cache = {}

    # Create interpolants for the history of our galaxy
    # Since there is no history yet, just give the initial value for everything
    def gas_hist(t):
        if hasattr(t, "__len__"):
            return np.array([init_gas] * len(t))
        else:
            return np.array(init_gas)

    def star_hist(t):
        if hasattr(t, "__len__"):
            return np.array([init_star] * len(t))
        else:
            return np.array(init_star)

    def metal_hist(t):
        if hasattr(t, "__len__"):
            return np.array([init_metal] * len(t))
        else:
            return np.array(init_metal)

    def dust_hist(t):
        if hasattr(t, "__len__"):
            return np.array([init_dust] * len(t))
        else:
            return np.array(init_dust)

    redshift = z_at_t(t)

    interp = [gas_hist, star_hist, metal_hist, dust_hist, None]

    sfr = sfr_model(model_params, t, redshift, *y1, *interp[:-1], cache)

    star_formation_rates[0] = sfr

    def sfr_hist(t):
        if hasattr(t, "__len__"):
            return np.array([sfr] * len(t))
        else:
            return np.array(sfr)

    interp[-1] = sfr_hist

    # Keep track of how many steps were attempted. If a large number
    # of steps fail, that could mean that the problem is stiff
    # or that there are breaking points causing problems.
    # Switch to an implicit method in this case.
    cache["attempted_steps"] = 0

    while t < time_end:

        # calculate all derivatives at current time
        dmgas_astration = sfr * (mgas / mgas[0])
        dmmetal_astration = sfr * (mmetal / mgas[0])
        dmdust_astration = sfr * (mdust / mgas[0])

        dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
            model_params, sfr, imf, t, redshift, *y1, *interp, cache
        )

        dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
            model_params, sfr, imf, t, redshift, *y1, *interp, cache
        )

        dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
            model_params, sfr, imf, t, redshift, *y1, *interp, cache
        )

        dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
            model_params, sfr, imf, t, redshift, *y1, *interp, cache
        )

        dmdust_grain_growth = grain_growth_model(
            model_params, sfr, imf, t, redshift, *y1, *interp, cache
        )

        dmdust_destruction = destruction_model(
            model_params, sfr, imf, t, redshift, *y1, *interp, cache
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

        # these derivs won't change, so store them in the output
        dgas_masses[i] = dmgas
        dstar_masses[i] = dmstars
        dmetal_masses[i] = dmmetal
        ddust_masses[i] = dmdust

        while True:

            # find out where our mini-step takes place
            mgas_int = mgas + dmgas * dt
            mstar_int = mstar + dmstars * dt
            mmetal_int = mmetal + dmmetal * dt
            mdust_int = mdust + dmdust * dt
            y2 = [mgas_int, mstar_int, mmetal_int, mdust_int]

            # Remake interp functions updated with derivs from the prev stage
            # and the loc of mini-step. Guess that the deriv at the current
            # mini step is equal to the deriv at the actual step.
            # TODO: This is extremely unrigorous. Figure out how to improve.
            # Maybe check the difference between the guess and what we calc,
            # and do a secant correction or something if that diff is too high,
            # Otherwise look at diff intep schema or perhaps even to a diff int
            # method.
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

            sfr_int = sfr_model(model_params, t, redshift,
                                *y2, *interp[:-1], cache)

            sfr_hist = CubicSpline(
                np.append(times[: i + 1], t + dt),
                np.append(star_formation_rates[: i + 1], sfr_int),
            )

            interp = [gas_hist, star_hist, metal_hist, dust_hist, sfr_hist]

            # calculate derivatives at the current mini-step
            dmgas_astration = sfr_int * (mgas_int / mgas_int[0])
            dmmetal_astration = sfr_int * (mmetal_int / mgas_int[0])
            dmdust_astration = sfr_int * (mdust_int / mgas_int[0])

            dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
                model_params, sfr_int, imf, t + dt, redshift, *y2, *interp, cache
            )

            dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
                model_params, sfr_int, imf, t + dt, redshift, *y2, *interp, cache
            )

            dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
                model_params, sfr_int, imf, t + dt, redshift, *y2, *interp, cache
            )

            dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
                model_params, sfr_int, imf, t + dt, redshift, *y2, *interp, cache
            )

            dmdust_grain_growth = grain_growth_model(
                model_params, sfr_int, imf, t + dt, redshift, *y2, *interp, cache
            )

            dmdust_destruction = destruction_model(
                model_params, sfr_int, imf, t + dt, redshift, *y2, *interp, cache
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

            # Using a linear comb of derivs at the current time and
            # our mini step, get both a prediction for the next timestep
            # and an estimate for the error on this timestep.
            mgas_fin = mgas + dt * (dmgas + dmgas_int) / 2
            mstar_fin = mstar + dt * (dmstars + dmstars_int) / 2
            mmetal_fin = mmetal + dt * (dmmetal + dmmetal_int) / 2
            mdust_fin = mdust + dt * (dmdust + dmdust_int) / 2

            err_gas = abs(dt * (-dmgas + dmgas_int) / 2)
            err_star = abs(dt * (-dmstars + dmstars_int) / 2)
            err_metal = abs(dt * (-dmmetal + dmmetal_int) / 2)
            err_dust = abs(dt * (-dmdust + dmdust_int) / 2)

            # Condense the errors into one RMS error value, weighted inversely
            # by the tolerance in that component (abs + rel)
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

            # whether the step is accepted or not, we modify the step size
            # in the hopes of 0 rejections and in keeping error at 90%
            # of our tolerance.
            dt = dt * max(0.5, min(2.0, 0.9 * (np.sqrt(1 / err))))

            # if the time step becomes extremely small, errstop
            # as we can reach a point where t + dt is identical to t
            # floating point shenaniganery :/
            if times[i + 1] - times[i] <= 1.1102230246251565e-14:
                raise RuntimeError(
                    "Stepsize too small for 64-bit precision. Either increasing or decreasing tolerance can help, though increasing is more likely."
                )

            # if the error is within our tolerance, accept the step
            # otherwise, restart from the first mini-step
            # (The derivs at the very start don't depend on dt)
            if err <= 1:
                break

        # If the space we've reserved for the output isn't enough,
        # reserve more space for the outputs
        if i + 2 >= len(times):

            n = len(times)
            times = np.append(times, np.full(n, np.inf))
            gas_masses = np.vstack((gas_masses, fp_zeros((n, len(init_gas)))))
            star_masses = np.vstack(
                (star_masses, fp_zeros((n, len(init_star)))))
            metal_masses = np.vstack(
                (metal_masses, fp_zeros((n, len(init_metal)))))
            dust_masses = np.vstack(
                (dust_masses, fp_zeros((n, len(init_dust)))))

            dgas_masses = np.vstack(
                (dgas_masses, fp_zeros((n, len(init_gas)))))
            dstar_masses = np.vstack(
                (dstar_masses, fp_zeros((n, len(init_star)))))
            dmetal_masses = np.vstack(
                (dmetal_masses, fp_zeros((n, len(init_metal)))))
            ddust_masses = np.vstack(
                (ddust_masses, fp_zeros((n, len(init_dust)))))

            star_formation_rates = np.append(star_formation_rates, fp_zeros(n))

        # set the accepted endpoint as the starting point for the next step
        t = times[i + 1]
        mgas = mgas_fin
        mstar = mstar_fin
        mmetal = mmetal_fin
        mdust = mdust_fin
        y1 = [mgas, mstar, mmetal, mdust]

        gas_masses[i + 1] = mgas
        star_masses[i + 1] = mstar
        metal_masses[i + 1] = mmetal
        dust_masses[i + 1] = mdust

        i += 1

        # remake the interp functions, again just doubling the last derivative
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

        sfr = sfr_model(model_params, t, redshift, *y1, *interp[:-1], cache)

        star_formation_rates[i] = sfr

        sfr_hist = CubicSpline(
            times[: i + 1],
            star_formation_rates[: i + 1],
        )

        interp = [gas_hist, star_hist, metal_hist, dust_hist, sfr_hist]

        # Nice little progress bar
        update_progress(t / time_end)

    # calculate the derivatives at the very end, so that we can Hermite int.
    dmgas_astration = sfr * (mgas / mgas[0])
    dmmetal_astration = sfr * (mmetal / mgas[0])
    dmdust_astration = sfr * (mdust / mgas[0])

    dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
        model_params, sfr, imf, t, redshift, *y1, *interp, cache
    )

    dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
        model_params, sfr, imf, t, redshift, *y1, *interp, cache
    )

    dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
        model_params, sfr, imf, t, redshift, *y1, *interp, cache
    )

    dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
        model_params, sfr, imf, t, redshift, *y1, *interp, cache
    )

    dmdust_grain_growth = grain_growth_model(
        model_params, sfr, imf, t, redshift, *y1, *interp, cache
    )

    dmdust_destruction = destruction_model(
        model_params, sfr, imf, t, redshift, *y1, *interp, cache
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

    # any part of the array that hasn't been filled in
    # gets discarded
    to_keep = times != np.inf

    # package everything up into the results array
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


def evolve_3o(
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
    dt = 0.003
    steps_guess = int(np.ceil((time_end - time_start) / dt))

    # Create arrays to store all outputs
    # The derivatives are needed to create hermite interpolants for the output
    # so we can get a good guess as to outputs even where our sim doesn't visit.
    # Hermite int. has error order O(h^4), while our method only has O(h^3),
    # so not really any extra error introduced by interpolating.
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

    # Higher order methods have "mini-steps," so init arrays to store those
    mgas_int1 = np.array(init_gas, dtype=fp)
    mstar_int1 = np.array(init_star, dtype=fp)
    mmetal_int1 = np.array(init_metal, dtype=fp)
    mdust_int1 = np.array(init_dust, dtype=fp)
    mgas_int2 = np.array(init_gas, dtype=fp)
    mstar_int2 = np.array(init_star, dtype=fp)
    mmetal_int2 = np.array(init_metal, dtype=fp)
    mdust_int2 = np.array(init_dust, dtype=fp)
    mgas_int3 = np.array(init_gas, dtype=fp)
    mstar_int3 = np.array(init_star, dtype=fp)
    mmetal_int3 = np.array(init_metal, dtype=fp)
    mdust_int3 = np.array(init_dust, dtype=fp)

    # package each step for ease of reading in below function calls
    # takes it from 17 lines per call to 3
    y1 = [mgas, mstar, mmetal, mdust]
    y2 = [mgas_int1, mstar_int1, mmetal_int1, mdust_int1]
    y3 = [mgas_int2, mstar_int2, mmetal_int2, mdust_int2]
    y4 = [mgas_int3, mstar_int3, mmetal_int3, mdust_int3]

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

    # Keep track of what step we're on, and also provide
    # a dictionary that lets functions give additional output/
    # store certain results in memory to speed up later computation.
    i = 0
    cache = {}

    # Create interpolants for the history of our galaxy
    # Since there is no history yet, just give the initial value for everything
    def gas_hist(t):
        if hasattr(t, "__len__"):
            return np.array([init_gas] * len(t))
        else:
            return np.array(init_gas)

    def star_hist(t):
        if hasattr(t, "__len__"):
            return np.array([init_star] * len(t))
        else:
            return np.array(init_star)

    def metal_hist(t):
        if hasattr(t, "__len__"):
            return np.array([init_metal] * len(t))
        else:
            return np.array(init_metal)

    def dust_hist(t):
        if hasattr(t, "__len__"):
            return np.array([init_dust] * len(t))
        else:
            return np.array(init_dust)

    redshift = z_at_t(t)

    interp = [gas_hist, star_hist, metal_hist, dust_hist, None]

    sfr = sfr_model(model_params, t, redshift, *y1, *interp[:-1], cache)

    star_formation_rates[0] = sfr

    def sfr_hist(t):
        if hasattr(t, "__len__"):
            return np.array([sfr] * len(t))
        else:
            return np.array(sfr)

    interp[-1] = sfr_hist

    # Keep track of how many steps were attempted. If a large number
    # of steps fail, that could mean that the problem is stiff
    # or that there are breaking points causing problems.
    # Switch to an implicit method in this case.
    cache["attempted_steps"] = 0

    while t < time_end:

        # calculate all derivatives at current time
        dmgas_astration = sfr * (mgas / mgas[0])
        dmmetal_astration = sfr * (mmetal / mgas[0])
        dmdust_astration = sfr * (mdust / mgas[0])

        dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
            model_params, sfr, imf, t, redshift, *y1, *interp, cache
        )

        dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
            model_params, sfr, imf, t, redshift, *y1, *interp, cache
        )

        dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
            model_params, sfr, imf, t, redshift, *y1, *interp, cache
        )

        dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
            model_params, sfr, imf, t, redshift, *y1, *interp, cache
        )

        dmdust_grain_growth = grain_growth_model(
            model_params, sfr, imf, t, redshift, *y1, *interp, cache
        )

        dmdust_destruction = destruction_model(
            model_params, sfr, imf, t, redshift, *y1, *interp, cache
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

        # these derivs won't change, so store them in the output
        dgas_masses[i] = dmgas
        dstar_masses[i] = dmstars
        dmetal_masses[i] = dmmetal
        ddust_masses[i] = dmdust

        while True:

            # find out where our mini-step takes place
            mgas_int1 = mgas + dmgas * (dt / 2)
            mstar_int1 = mstar + dmstars * (dt / 2)
            mmetal_int1 = mmetal + dmmetal * (dt / 2)
            mdust_int1 = mdust + dmdust * (dt / 2)
            y2 = [mgas_int1, mstar_int1, mmetal_int1, mdust_int1]

            # Remake interp functions updated with derivs from the prev stage
            # and the loc of mini-step. Guess that the deriv at the current
            # mini step is equal to the deriv at the actual step.
            # TODO: This is extremely unrigorous. Figure out how to improve.
            # Maybe check the difference between the guess and what we calc,
            # and do a secant correction or something if that diff is too high,
            # Otherwise look at diff intep schema or perhaps even to a diff int
            # method.
            gas_hist = CubicHermiteSpline(
                np.append(times[: i + 1], t + dt),
                np.vstack((gas_masses[: i + 1], mgas_int1)),
                np.vstack((dgas_masses[: i + 1], dgas_masses[i])),
            )
            star_hist = CubicHermiteSpline(
                np.append(times[: i + 1], t + dt),
                np.vstack((star_masses[: i + 1], mstar_int1)),
                np.vstack((dstar_masses[: i + 1], dstar_masses[i])),
            )
            metal_hist = CubicHermiteSpline(
                np.append(times[: i + 1], t + dt),
                np.vstack((metal_masses[: i + 1], mmetal_int1)),
                np.vstack((dmetal_masses[: i + 1], dmetal_masses[i])),
            )
            dust_hist = CubicHermiteSpline(
                np.append(times[: i + 1], t + dt),
                np.vstack((dust_masses[: i + 1], mdust_int1)),
                np.vstack((ddust_masses[: i + 1], ddust_masses[i])),
            )

            redshift = z_at_t(t + dt)

            sfr_int1 = sfr_model(model_params, t, redshift,
                                 *y2, *interp[:-1], cache)

            sfr_hist = CubicSpline(
                np.append(times[: i + 1], t + dt),
                np.append(star_formation_rates[: i + 1], sfr_int1),
            )

            interp = [gas_hist, star_hist, metal_hist, dust_hist, sfr_hist]

            # calculate derivatives at the current mini-step
            dmgas_astration = sfr_int1 * (mgas_int1 / mgas_int1[0])
            dmmetal_astration = sfr_int1 * (mmetal_int1 / mgas_int1[0])
            dmdust_astration = sfr_int1 * (mdust_int1 / mgas_int1[0])

            dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
                model_params, sfr_int1, imf, t + dt, redshift, *y2, *interp, cache
            )

            dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
                model_params, sfr_int1, imf, t + dt, redshift, *y2, *interp, cache
            )

            dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
                model_params, sfr_int1, imf, t + dt, redshift, *y2, *interp, cache
            )

            dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
                model_params, sfr_int1, imf, t + dt, redshift, *y2, *interp, cache
            )

            dmdust_grain_growth = grain_growth_model(
                model_params, sfr_int1, imf, t + dt, redshift, *y2, *interp, cache
            )

            dmdust_destruction = destruction_model(
                model_params, sfr_int1, imf, t + dt, redshift, *y2, *interp, cache
            )

            dmgas_int1 = (
                -dmgas_astration
                + dmgas_inflows
                - dmgas_outflows
                + dmgas_recycling
                + dmgas_ejecta
            )
            dmstars_int1 = sfr_int1 - dmgas_ejecta
            dmmetal_int1 = (
                -dmmetal_astration
                + dmmetal_inflows
                - dmmetal_outflows
                + dmmetal_recycling
                + dmmetal_ejecta
            )
            dmdust_int1 = (
                -dmdust_astration
                + dmdust_inflows
                - dmdust_outflows
                + dmdust_recycling
                + dmdust_ejecta
                + dmdust_grain_growth
                - dmdust_destruction
            )

            # find out where our mini-step takes place
            mgas_int2 = mgas + dmgas_int1 * (dt / 2)
            mstar_int2 = mstar + dmstars_int1 * (dt / 2)
            mmetal_int2 = mmetal + dmmetal_int1 * (dt / 2)
            mdust_int2 = mdust + dmdust_int1 * (dt / 2)
            y3 = [mgas_int2, mstar_int2, mmetal_int2, mdust_int2]

            # Remake interp functions updated with derivs from the prev stage
            # and the loc of mini-step. Guess that the deriv at the current
            # mini step is equal to the deriv at the actual step.
            # TODO: This is extremely unrigorous. Figure out how to improve.
            # Maybe check the difference between the guess and what we calc,
            # and do a secant correction or something if that diff is too high,
            # Otherwise look at diff intep schema or perhaps even to a diff int
            # method.
            gas_hist = CubicHermiteSpline(
                np.append(times[: i + 1], t + dt),
                np.vstack((gas_masses[: i + 1], mgas_int2)),
                np.vstack((dgas_masses[: i + 1], dgas_masses[i])),
            )
            star_hist = CubicHermiteSpline(
                np.append(times[: i + 1], t + dt),
                np.vstack((star_masses[: i + 1], mstar_int2)),
                np.vstack((dstar_masses[: i + 1], dstar_masses[i])),
            )
            metal_hist = CubicHermiteSpline(
                np.append(times[: i + 1], t + dt),
                np.vstack((metal_masses[: i + 1], mmetal_int2)),
                np.vstack((dmetal_masses[: i + 1], dmetal_masses[i])),
            )
            dust_hist = CubicHermiteSpline(
                np.append(times[: i + 1], t + dt),
                np.vstack((dust_masses[: i + 1], mdust_int2)),
                np.vstack((ddust_masses[: i + 1], ddust_masses[i])),
            )

            redshift = z_at_t(t + dt)

            sfr_int2 = sfr_model(model_params, t, redshift,
                                 *y3, *interp[:-1], cache)

            sfr_hist = CubicSpline(
                np.append(times[: i + 1], t + dt),
                np.append(star_formation_rates[: i + 1], sfr_int2),
            )

            interp = [gas_hist, star_hist, metal_hist, dust_hist, sfr_hist]

            # calculate derivatives at the current mini-step
            dmgas_astration = sfr_int2 * (mgas_int2 / mgas_int2[0])
            dmmetal_astration = sfr_int2 * (mmetal_int2 / mgas_int2[0])
            dmdust_astration = sfr_int2 * (mdust_int2 / mgas_int2[0])

            dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
                model_params, sfr_int2, imf, t + dt, redshift, *y3, *interp, cache
            )

            dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
                model_params, sfr_int2, imf, t + dt, redshift, *y3, *interp, cache
            )

            dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
                model_params, sfr_int2, imf, t + dt, redshift, *y3, *interp, cache
            )

            dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
                model_params, sfr_int2, imf, t + dt, redshift, *y3, *interp, cache
            )

            dmdust_grain_growth = grain_growth_model(
                model_params, sfr_int2, imf, t + dt, redshift, *y3, *interp, cache
            )

            dmdust_destruction = destruction_model(
                model_params, sfr_int2, imf, t + dt, redshift, *y3, *interp, cache
            )

            dmgas_int2 = (
                -dmgas_astration
                + dmgas_inflows
                - dmgas_outflows
                + dmgas_recycling
                + dmgas_ejecta
            )
            dmstars_int2 = sfr_int2 - dmgas_ejecta
            dmmetal_int2 = (
                -dmmetal_astration
                + dmmetal_inflows
                - dmmetal_outflows
                + dmmetal_recycling
                + dmmetal_ejecta
            )
            dmdust_int2 = (
                -dmdust_astration
                + dmdust_inflows
                - dmdust_outflows
                + dmdust_recycling
                + dmdust_ejecta
                + dmdust_grain_growth
                - dmdust_destruction
            )

            # find out where our mini-step takes place
            mgas_int3 = mgas + dmgas_int2 * dt
            mstar_int3 = mstar + dmstars_int2 * dt
            mmetal_int3 = mmetal + dmmetal_int2 * dt
            mdust_int3 = mdust + dmdust_int2 * dt
            y4 = [mgas_int3, mstar_int3, mmetal_int3, mdust_int3]

            # Remake interp functions updated with derivs from the prev stage
            # and the loc of mini-step. Guess that the deriv at the current
            # mini step is equal to the deriv at the actual step.
            # TODO: This is extremely unrigorous. Figure out how to improve.
            # Maybe check the difference between the guess and what we calc,
            # and do a secant correction or something if that diff is too high,
            # Otherwise look at diff intep schema or perhaps even to a diff int
            # method.
            gas_hist = CubicHermiteSpline(
                np.append(times[: i + 1], t + dt),
                np.vstack((gas_masses[: i + 1], mgas_int3)),
                np.vstack((dgas_masses[: i + 1], dgas_masses[i])),
            )
            star_hist = CubicHermiteSpline(
                np.append(times[: i + 1], t + dt),
                np.vstack((star_masses[: i + 1], mstar_int3)),
                np.vstack((dstar_masses[: i + 1], dstar_masses[i])),
            )
            metal_hist = CubicHermiteSpline(
                np.append(times[: i + 1], t + dt),
                np.vstack((metal_masses[: i + 1], mmetal_int3)),
                np.vstack((dmetal_masses[: i + 1], dmetal_masses[i])),
            )
            dust_hist = CubicHermiteSpline(
                np.append(times[: i + 1], t + dt),
                np.vstack((dust_masses[: i + 1], mdust_int3)),
                np.vstack((ddust_masses[: i + 1], ddust_masses[i])),
            )

            redshift = z_at_t(t + dt)

            sfr_int3 = sfr_model(model_params, t, redshift,
                                 *y4, *interp[:-1], cache)

            sfr_hist = CubicSpline(
                np.append(times[: i + 1], t + dt),
                np.append(star_formation_rates[: i + 1], sfr_int3),
            )

            interp = [gas_hist, star_hist, metal_hist, dust_hist, sfr_hist]

            # calculate derivatives at the current mini-step
            dmgas_astration = sfr_int3 * (mgas_int3 / mgas_int3[0])
            dmmetal_astration = sfr_int3 * (mmetal_int3 / mgas_int3[0])
            dmdust_astration = sfr_int3 * (mdust_int3 / mgas_int3[0])

            dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
                model_params, sfr_int3, imf, t + dt, redshift, *y4, *interp, cache
            )

            dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
                model_params, sfr_int3, imf, t + dt, redshift, *y4, *interp, cache
            )

            dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
                model_params, sfr_int3, imf, t + dt, redshift, *y4, *interp, cache
            )

            dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
                model_params, sfr_int3, imf, t + dt, redshift, *y4, *interp, cache
            )

            dmdust_grain_growth = grain_growth_model(
                model_params, sfr_int3, imf, t + dt, redshift, *y4, *interp, cache
            )

            dmdust_destruction = destruction_model(
                model_params, sfr_int3, imf, t + dt, redshift, *y4, *interp, cache
            )

            dmgas_int3 = (
                -dmgas_astration
                + dmgas_inflows
                - dmgas_outflows
                + dmgas_recycling
                + dmgas_ejecta
            )
            dmstars_int3 = sfr_int3 - dmgas_ejecta
            dmmetal_int3 = (
                -dmmetal_astration
                + dmmetal_inflows
                - dmmetal_outflows
                + dmmetal_recycling
                + dmmetal_ejecta
            )
            dmdust_int3 = (
                -dmdust_astration
                + dmdust_inflows
                - dmdust_outflows
                + dmdust_recycling
                + dmdust_ejecta
                + dmdust_grain_growth
                - dmdust_destruction
            )

            # Using a linear comb of derivs at the current time and
            # our mini step, get both a prediction for the next timestep
            # and an estimate for the error on this timestep.
            mgas_fin = (
                mgas + dt * (dmgas / 2 + dmgas_int1 +
                             dmgas_int2 + dmgas_int3 / 2) / 3
            )
            mstar_fin = (
                mstar
                + dt
                * (dmstars / 2 + dmstars_int1 + dmstars_int2 + dmstars_int3 / 2)
                / 3
            )
            mmetal_fin = (
                mmetal
                + dt
                * (dmmetal / 2 + dmmetal_int1 + dmmetal_int2 + dmmetal_int3 / 2)
                / 3
            )
            mdust_fin = (
                mdust
                + dt * (dmdust / 2 + dmdust_int1 +
                        dmdust_int2 + dmdust_int3 / 2) / 3
            )

            err_gas = (
                dt * (dmgas / 2 + dmgas_int1 - 2 *
                      dmgas_int2 + dmgas_int3 / 2) / 3
            )
            err_star = (
                dt
                * (dmstars / 2 + dmstars_int1 - 2 * dmstars_int2 + dmstars_int3 / 2)
                / 3
            )
            err_metal = (
                dt
                * (dmmetal / 2 + dmmetal_int1 - 2 * dmmetal_int2 + dmmetal_int3 / 2)
                / 3
            )
            err_dust = (
                dt * (dmdust / 2 + dmdust_int1 - 2 *
                      dmdust_int2 + dmdust_int3 / 2) / 3
            )

            # Condense the errors into one RMS error value, weighted inversely
            # by the tolerance in that component (abs + rel)
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

            # whether the step is accepted or not, we modify the step size
            # in the hopes of 0 rejections and in keeping error at 90%
            # of our tolerance.
            dt = dt * max(0.5, min(2.0, 0.9 * (np.sqrt(1 / err))))

            # if the time step becomes extremely small, errstop
            # as we can reach a point where t + dt is identical to t
            # floating point shenaniganery :/
            if times[i + 1] - times[i] <= 1.1102230246251565e-14:
                raise RuntimeError(
                    "Stepsize too small for 64-bit precision. Either increasing or decreasing tolerance can help, though increasing is more likely."
                )

            # if the error is within our tolerance, accept the step
            # otherwise, restart from the first mini-step
            # (The derivs at the very start don't depend on dt)
            if err <= 1:
                break

        # If the space we've reserved for the output isn't enough,
        # reserve more space for the outputs
        if i + 2 >= len(times):

            n = len(times)
            times = np.append(times, np.full(n, np.inf))
            gas_masses = np.vstack((gas_masses, fp_zeros((n, len(init_gas)))))
            star_masses = np.vstack(
                (star_masses, fp_zeros((n, len(init_star)))))
            metal_masses = np.vstack(
                (metal_masses, fp_zeros((n, len(init_metal)))))
            dust_masses = np.vstack(
                (dust_masses, fp_zeros((n, len(init_dust)))))

            dgas_masses = np.vstack(
                (dgas_masses, fp_zeros((n, len(init_gas)))))
            dstar_masses = np.vstack(
                (dstar_masses, fp_zeros((n, len(init_star)))))
            dmetal_masses = np.vstack(
                (dmetal_masses, fp_zeros((n, len(init_metal)))))
            ddust_masses = np.vstack(
                (ddust_masses, fp_zeros((n, len(init_dust)))))

            star_formation_rates = np.append(star_formation_rates, fp_zeros(n))

        # set the accepted endpoint as the starting point for the next step
        t = times[i + 1]
        mgas = mgas_fin
        mstar = mstar_fin
        mmetal = mmetal_fin
        mdust = mdust_fin
        y1 = [mgas, mstar, mmetal, mdust]

        gas_masses[i + 1] = mgas
        star_masses[i + 1] = mstar
        metal_masses[i + 1] = mmetal
        dust_masses[i + 1] = mdust

        i += 1

        # remake the interp functions, again just doubling the last derivative
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

        sfr = sfr_model(model_params, t, redshift, *y1, *interp[:-1], cache)

        star_formation_rates[i] = sfr

        sfr_hist = CubicSpline(
            times[: i + 1],
            star_formation_rates[: i + 1],
        )

        interp = [gas_hist, star_hist, metal_hist, dust_hist, sfr_hist]

        # Nice little progress bar
        update_progress(t / time_end)

    # calculate the derivatives at the very end, so that we can Hermite int.
    dmgas_astration = sfr * (mgas / mgas[0])
    dmmetal_astration = sfr * (mmetal / mgas[0])
    dmdust_astration = sfr * (mdust / mgas[0])

    dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
        model_params, sfr, imf, t, redshift, *y1, *interp, cache
    )

    dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
        model_params, sfr, imf, t, redshift, *y1, *interp, cache
    )

    dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
        model_params, sfr, imf, t, redshift, *y1, *interp, cache
    )

    dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
        model_params, sfr, imf, t, redshift, *y1, *interp, cache
    )

    dmdust_grain_growth = grain_growth_model(
        model_params, sfr, imf, t, redshift, *y1, *interp, cache
    )

    dmdust_destruction = destruction_model(
        model_params, sfr, imf, t, redshift, *y1, *interp, cache
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

    # any part of the array that hasn't been filled in
    # gets discarded
    to_keep = times != np.inf

    # package everything up into the results array
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


def update_progress(progress):
    """
    from user Brian Khuu on stack exchange, displays a nice little
    progress bar.
    """
    barLength = 50  # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength * progress))
    text = "\rPercent: [{0}] {1:.0f}% {2}".format(
        "█" * block + "-" * (barLength - block), progress * 100, status
    )
    sys.stdout.write(text)
    sys.stdout.flush()
