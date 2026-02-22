import numpy as np
from dustdevol.generic import fp, fp_zeros, z_at_t
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

    err_order = 1  # order of error scaling of the lower order method
    safety_factor = 0.9  # will try to keep error at this % of tolerance
    max_increase = 2  # stepsize increases by at most this factor at one time
    max_decrease = 0.5  # stepsize decreases by at most this factor at one time

    # Components of the butcher tableaux for the method, in short
    # each row of tableuax_a defines a stage of the method, and what
    # combination of derivs from each stage to use for that stage
    # must be strictly lower triangular
    # each component of tableaux_c tells what multiple of the stepsize to use
    # for the time of that stage
    # each component of tableaux_b defines the weight with which the deriv from
    # that stage contributes to the final answer to be propagated
    # each component of tableaux_d defines the weight with which the deriv from
    # that stage contributes to error estimate
    tableaux_a = np.array(((0, 0), (1, 0)))
    tableaux_b = np.array((1 / 2, 1 / 2))
    tableaux_c = np.array((0, 1))
    tableaux_d = np.array((-1 / 2, 1 / 2))
    stages = len(tableaux_c)

    # guess at needed time step, and allocate space assuming that's
    # the time step.
    # TODO: More sophisticated first step choice
    dt = 0.003
    steps_guess = int(np.ceil((time_end - time_start) / dt))

    # Create arrays to store all outputs
    # The derivatives are needed to create hermite interpolants for the output
    # so we can get a good guess as to outputs even where our sim doesn't visit
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
    mgas_int = np.array([init_gas for i in range(stages)], dtype=fp)
    mstar_int = np.array([init_star for i in range(stages)], dtype=fp)
    mmetal_int = np.array([init_metal for i in range(stages)], dtype=fp)
    mdust_int = np.array([init_dust for i in range(stages)], dtype=fp)

    # package each step for ease of reading in below function calls
    # takes it from 17 lines per call to 3
    y = [mgas_int[0], mstar_int[0], mmetal_int[0], mdust_int[0]]

    # initialize the variables to hold the change in each population due to
    # different phenomena, namely star formation, inflows, outflows,
    # outflow recycling, ejecta from dying (or perhaps still living) stars,
    # as well as grain growth and destruction in the ISM and clouds
    dmgas_int = fp_zeros((stages, len(init_gas)))
    dmstar_int = fp_zeros((stages, len(init_star)))
    dmmetal_int = fp_zeros((stages, len(init_metal)))
    dmdust_int = fp_zeros((stages, len(init_dust)))

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

    sfr = sfr_model(model_params, t, redshift, *y, *interp[:-1], cache)

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

        restart_point = 0
        accepted = False

        while not accepted:
            for j in range(restart_point, stages):

                restart_point = 1
                step = tableaux_c[j] * dt

                # find out where our mini-step takes place
                mgas_int[j] = mgas_int[0] + (
                    dmgas_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                mstar_int[j] = mstar_int[0] + (
                    dmstar_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                mmetal_int[j] = mmetal_int[0] + (
                    dmmetal_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                mdust_int[j] = mdust_int[0] + (
                    dmdust_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                y = [
                    mgas_int[j],
                    mstar_int[j],
                    mmetal_int[j],
                    mdust_int[j],
                ]

                # Remake interp functions updated with derivs from the prev stage
                # and the loc of mini-step. Guess that the deriv at the current
                # mini step is equal to the deriv at the actual step.
                # TODO: This is extremely unrigorous. Figure out how to improve.
                # Maybe check the difference between the guess and what we calc,
                # and do a secant correction or something if that diff is too high,
                # Otherwise look at diff intep schema or perhaps even to a diff int
                # method.
                try:
                    gas_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((gas_masses[: i + 1], mgas_int[j])),
                        np.vstack((dgas_masses[: i + 1], dgas_masses[i])),
                    )
                    star_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((star_masses[: i + 1], mstar_int[j])),
                        np.vstack((dstar_masses[: i + 1], dstar_masses[i])),
                    )
                    metal_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((metal_masses[: i + 1], mmetal_int[j])),
                        np.vstack((dmetal_masses[: i + 1], dmetal_masses[i])),
                    )
                    dust_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((dust_masses[: i + 1], mdust_int[j])),
                        np.vstack((ddust_masses[: i + 1], ddust_masses[i])),
                    )

                    redshift = z_at_t(t + step)

                    sfr = sfr_model(
                        model_params, t + step, redshift, *
                        y, *interp[:-1], cache
                    )

                    sfr_hist = CubicSpline(
                        np.append(times[: i + 1], t + step),
                        np.append(star_formation_rates[: i + 1], sfr),
                    )

                    interp = [gas_hist, star_hist,
                              metal_hist, dust_hist, sfr_hist]

                except ValueError:
                    pass

                # calculate all derivatives at current time
                dmgas_astration = sfr * (mgas_int[j] / mgas_int[j, 0])
                dmmetal_astration = sfr * (mmetal_int[j] / mgas_int[j, 0])
                dmdust_astration = sfr * (mdust_int[j] / mgas_int[j, 0])

                dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmdust_grain_growth = grain_growth_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmdust_destruction = destruction_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_int[j] = (
                    -dmgas_astration
                    + dmgas_inflows
                    - dmgas_outflows
                    + dmgas_recycling
                    + dmgas_ejecta
                )
                dmstar_int[j] = sfr - dmgas_ejecta
                dmmetal_int[j] = (
                    -dmmetal_astration
                    + dmmetal_inflows
                    - dmmetal_outflows
                    + dmmetal_recycling
                    + dmmetal_ejecta
                )
                dmdust_int[j] = (
                    -dmdust_astration
                    + dmdust_inflows
                    - dmdust_outflows
                    + dmdust_recycling
                    + dmdust_ejecta
                    + dmdust_grain_growth
                    - dmdust_destruction
                )

                if j == 0:
                    # these derivs won't change, so store them in the output
                    dgas_masses[i] = dmgas_int[j]
                    dstar_masses[i] = dmstar_int[j]
                    dmetal_masses[i] = dmmetal_int[j]
                    ddust_masses[i] = dmdust_int[j]
                    star_formation_rates[i] = sfr

            # Using a linear comb of derivs at the current time and
            # our mini step, get both a prediction for the next timestep
            # and an estimate for the error on this timestep.
            mgas_fin = mgas_int[0] + (dmgas_int.transpose() * tableaux_b * dt).sum(
                axis=1
            )
            mstar_fin = mstar_int[0] + (dmstar_int.transpose() * tableaux_b * dt).sum(
                axis=1
            )
            mmetal_fin = mmetal_int[0] + (
                dmmetal_int.transpose() * tableaux_b * dt
            ).sum(axis=1)
            mdust_fin = mdust_int[0] + (dmdust_int.transpose() * tableaux_b * dt).sum(
                axis=1
            )

            err_gas = abs(
                (dmgas_int.transpose() * tableaux_d * dt).sum(axis=1))
            err_star = abs((dmstar_int.transpose() *
                           tableaux_d * dt).sum(axis=1))
            err_metal = abs((dmmetal_int.transpose() *
                            tableaux_d * dt).sum(axis=1))
            err_dust = abs((dmdust_int.transpose() *
                           tableaux_d * dt).sum(axis=1))

            # Condense the errors into one RMS error value, weighted inversely
            # by the tolerance in that component (abs + rel)
            err = np.hstack((err_gas, err_star, err_metal, err_dust)) / (
                absolute_tolerance
                + relative_tolerance
                * np.maximum(
                    np.hstack((mgas_fin, mstar_fin, mmetal_fin, mdust_fin)),
                    np.hstack((mgas_int[0], mstar_int[0],
                              mmetal_int[0], mdust_int[0])),
                )
            )

            err = np.sqrt((err**2).mean())

            times[i + 1] = t + dt

            cache["attempted_steps"] += 1

            # whether the step is accepted or not, we modify the step size
            # in the hopes of 0 rejections and in keeping error at 90%
            # of our tolerance.
            dt = dt * max(
                max_decrease,
                min(
                    max_increase,
                    safety_factor * (np.sqrt(1 / err) **
                                     (1 / (err_order + 1))),
                ),
            )

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
                accepted = True

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
        mgas_int[0] = mgas_fin
        mstar_int[0] = mstar_fin
        mmetal_int[0] = mmetal_fin
        mdust_int[0] = mdust_fin
        y = [mgas_int[0], mstar_int[0], mmetal_int[0], mdust_int[0]]

        gas_masses[i + 1] = mgas_int[0]
        star_masses[i + 1] = mstar_int[0]
        metal_masses[i + 1] = mmetal_int[0]
        dust_masses[i + 1] = mdust_int[0]

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

        sfr = sfr_model(model_params, t, redshift, *y, *interp[:-1], cache)

        star_formation_rates[i] = sfr

        sfr_hist = CubicSpline(
            times[: i + 1],
            star_formation_rates[: i + 1],
        )

        interp = [gas_hist, star_hist, metal_hist, dust_hist, sfr_hist]

        # Nice little progress bar
        update_progress(t / time_end)

    # calculate the derivatives at the very end, so that we can Hermite int.
    dmgas_astration = sfr * (mgas_int[0] / mgas_int[0, 0])
    dmmetal_astration = sfr * (mmetal_int[0] / mgas_int[0, 0])
    dmdust_astration = sfr * (mdust_int[0] / mgas_int[0, 0])

    dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmdust_grain_growth = grain_growth_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmdust_destruction = destruction_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
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

    err_order = 2  # order of error scaling of the lower order method
    safety_factor = 0.9  # will try to keep error at this % of tolerance
    max_increase = 2  # stepsize increases by at most this factor at one time
    max_decrease = 0.5  # stepsize decreases by at most this factor at one time

    # Components of the butcher tableaux for the method, in short
    # each row of tableuax_a defines a stage of the method, and what
    # combination of derivs from each stage to use for that stage
    # must be strictly lower triangular
    # each component of tableaux_c tells what multiple of the stepsize to use
    # for the time of that stage
    # each component of tableaux_b defines the weight with which the deriv from
    # that stage contributes to the final answer to be propagated
    # each component of tableaux_d defines the weight with which the deriv from
    # that stage contributes to error estimate
    tableaux_a = np.array(
        ((0, 0, 0, 0), (1 / 2, 0, 0, 0), (0, 1 / 2, 0, 0), (0, 0, 1, 0))
    )
    tableaux_b = np.array((1 / 6, 1 / 3, 1 / 3, 1 / 6))
    tableaux_c = np.array((0, 1 / 2, 1 / 2, 1))
    tableaux_d = np.array((1 / 6, 1 / 3, -2 / 3, 1 / 6))
    stages = len(tableaux_c)

    # guess at needed time step, and allocate space assuming that's
    # the time step.
    # TODO: More sophisticated first step choice
    dt = 0.003
    steps_guess = int(np.ceil((time_end - time_start) / dt))

    # Create arrays to store all outputs
    # The derivatives are needed to create hermite interpolants for the output
    # so we can get a good guess as to outputs even where our sim doesn't visit
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
    mgas_int = np.array([init_gas for i in range(stages)], dtype=fp)
    mstar_int = np.array([init_star for i in range(stages)], dtype=fp)
    mmetal_int = np.array([init_metal for i in range(stages)], dtype=fp)
    mdust_int = np.array([init_dust for i in range(stages)], dtype=fp)

    # package each step for ease of reading in below function calls
    # takes it from 17 lines per call to 3
    y = [mgas_int[0], mstar_int[0], mmetal_int[0], mdust_int[0]]

    # initialize the variables to hold the change in each population due to
    # different phenomena, namely star formation, inflows, outflows,
    # outflow recycling, ejecta from dying (or perhaps still living) stars,
    # as well as grain growth and destruction in the ISM and clouds
    dmgas_int = fp_zeros((stages, len(init_gas)))
    dmstar_int = fp_zeros((stages, len(init_star)))
    dmmetal_int = fp_zeros((stages, len(init_metal)))
    dmdust_int = fp_zeros((stages, len(init_dust)))

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

    sfr = sfr_model(model_params, t, redshift, *y, *interp[:-1], cache)

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

        restart_point = 0
        accepted = False

        while not accepted:
            for j in range(restart_point, stages):

                restart_point = 1
                step = tableaux_c[j] * dt

                # find out where our mini-step takes place
                mgas_int[j] = mgas_int[0] + (
                    dmgas_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                mstar_int[j] = mstar_int[0] + (
                    dmstar_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                mmetal_int[j] = mmetal_int[0] + (
                    dmmetal_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                mdust_int[j] = mdust_int[0] + (
                    dmdust_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                y = [
                    mgas_int[j],
                    mstar_int[j],
                    mmetal_int[j],
                    mdust_int[j],
                ]

                # Remake interp functions updated with derivs from the prev stage
                # and the loc of mini-step. Guess that the deriv at the current
                # mini step is equal to the deriv at the actual step.
                # TODO: This is extremely unrigorous. Figure out how to improve.
                # Maybe check the difference between the guess and what we calc,
                # and do a secant correction or something if that diff is too high,
                # Otherwise look at diff intep schema or perhaps even to a diff int
                # method.
                try:
                    gas_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((gas_masses[: i + 1], mgas_int[j])),
                        np.vstack((dgas_masses[: i + 1], dgas_masses[i])),
                    )
                    star_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((star_masses[: i + 1], mstar_int[j])),
                        np.vstack((dstar_masses[: i + 1], dstar_masses[i])),
                    )
                    metal_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((metal_masses[: i + 1], mmetal_int[j])),
                        np.vstack((dmetal_masses[: i + 1], dmetal_masses[i])),
                    )
                    dust_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((dust_masses[: i + 1], mdust_int[j])),
                        np.vstack((ddust_masses[: i + 1], ddust_masses[i])),
                    )

                    redshift = z_at_t(t + step)

                    sfr = sfr_model(
                        model_params, t + step, redshift, *
                        y, *interp[:-1], cache
                    )

                    sfr_hist = CubicSpline(
                        np.append(times[: i + 1], t + step),
                        np.append(star_formation_rates[: i + 1], sfr),
                    )

                    interp = [gas_hist, star_hist,
                              metal_hist, dust_hist, sfr_hist]

                except ValueError:
                    pass

                # calculate all derivatives at current time
                dmgas_astration = sfr * (mgas_int[j] / mgas_int[j, 0])
                dmmetal_astration = sfr * (mmetal_int[j] / mgas_int[j, 0])
                dmdust_astration = sfr * (mdust_int[j] / mgas_int[j, 0])

                dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmdust_grain_growth = grain_growth_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmdust_destruction = destruction_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_int[j] = (
                    -dmgas_astration
                    + dmgas_inflows
                    - dmgas_outflows
                    + dmgas_recycling
                    + dmgas_ejecta
                )
                dmstar_int[j] = sfr - dmgas_ejecta
                dmmetal_int[j] = (
                    -dmmetal_astration
                    + dmmetal_inflows
                    - dmmetal_outflows
                    + dmmetal_recycling
                    + dmmetal_ejecta
                )
                dmdust_int[j] = (
                    -dmdust_astration
                    + dmdust_inflows
                    - dmdust_outflows
                    + dmdust_recycling
                    + dmdust_ejecta
                    + dmdust_grain_growth
                    - dmdust_destruction
                )

                if j == 0:
                    # these derivs won't change, so store them in the output
                    dgas_masses[i] = dmgas_int[j]
                    dstar_masses[i] = dmstar_int[j]
                    dmetal_masses[i] = dmmetal_int[j]
                    ddust_masses[i] = dmdust_int[j]
                    star_formation_rates[i] = sfr

            # Using a linear comb of derivs at the current time and
            # our mini step, get both a prediction for the next timestep
            # and an estimate for the error on this timestep.
            mgas_fin = mgas_int[0] + (dmgas_int.transpose() * tableaux_b * dt).sum(
                axis=1
            )
            mstar_fin = mstar_int[0] + (dmstar_int.transpose() * tableaux_b * dt).sum(
                axis=1
            )
            mmetal_fin = mmetal_int[0] + (
                dmmetal_int.transpose() * tableaux_b * dt
            ).sum(axis=1)
            mdust_fin = mdust_int[0] + (dmdust_int.transpose() * tableaux_b * dt).sum(
                axis=1
            )

            err_gas = abs(
                (dmgas_int.transpose() * tableaux_d * dt).sum(axis=1))
            err_star = abs((dmstar_int.transpose() *
                           tableaux_d * dt).sum(axis=1))
            err_metal = abs((dmmetal_int.transpose() *
                            tableaux_d * dt).sum(axis=1))
            err_dust = abs((dmdust_int.transpose() *
                           tableaux_d * dt).sum(axis=1))

            # Condense the errors into one RMS error value, weighted inversely
            # by the tolerance in that component (abs + rel)
            err = np.hstack((err_gas, err_star, err_metal, err_dust)) / (
                absolute_tolerance
                + relative_tolerance
                * np.maximum(
                    np.hstack((mgas_fin, mstar_fin, mmetal_fin, mdust_fin)),
                    np.hstack((mgas_int[0], mstar_int[0],
                              mmetal_int[0], mdust_int[0])),
                )
            )

            err = np.sqrt((err**2).mean())

            times[i + 1] = t + dt

            cache["attempted_steps"] += 1

            # whether the step is accepted or not, we modify the step size
            # in the hopes of 0 rejections and in keeping error at 90%
            # of our tolerance.
            dt = dt * max(
                max_decrease,
                min(
                    max_increase,
                    safety_factor * (np.sqrt(1 / err) **
                                     (1 / (err_order + 1))),
                ),
            )

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
                accepted = True

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
        mgas_int[0] = mgas_fin
        mstar_int[0] = mstar_fin
        mmetal_int[0] = mmetal_fin
        mdust_int[0] = mdust_fin
        y = [mgas_int[0], mstar_int[0], mmetal_int[0], mdust_int[0]]

        gas_masses[i + 1] = mgas_int[0]
        star_masses[i + 1] = mstar_int[0]
        metal_masses[i + 1] = mmetal_int[0]
        dust_masses[i + 1] = mdust_int[0]

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

        sfr = sfr_model(model_params, t, redshift, *y, *interp[:-1], cache)

        star_formation_rates[i] = sfr

        sfr_hist = CubicSpline(
            times[: i + 1],
            star_formation_rates[: i + 1],
        )

        interp = [gas_hist, star_hist, metal_hist, dust_hist, sfr_hist]

        # Nice little progress bar
        update_progress(t / time_end)

    # calculate the derivatives at the very end, so that we can Hermite int.
    dmgas_astration = sfr * (mgas_int[0] / mgas_int[0, 0])
    dmmetal_astration = sfr * (mmetal_int[0] / mgas_int[0, 0])
    dmdust_astration = sfr * (mdust_int[0] / mgas_int[0, 0])

    dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmdust_grain_growth = grain_growth_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmdust_destruction = destruction_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
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


def evolve_4o(
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

    err_order = 3  # order of error scaling of the lower order method
    safety_factor = 0.9  # will try to keep error at this % of tolerance
    max_increase = 2  # stepsize increases by at most this factor at one time
    max_decrease = 0.5  # stepsize decreases by at most this factor at one time

    # Components of the butcher tableaux for the method, in short
    # each row of tableuax_a defines a stage of the method, and what
    # combination of derivs from each stage to use for that stage
    # must be strictly lower triangular
    # each component of tableaux_c tells what multiple of the stepsize to use
    # for the time of that stage
    # each component of tableaux_b defines the weight with which the deriv from
    # that stage contributes to the final answer to be propagated
    # each component of tableaux_d defines the weight with which the deriv from
    # that stage contributes to error estimate
    tableaux_a = np.array(
        ((0, 0, 0, 0, 0),
         (2/7, 0, 0, 0, 0),
         (-8/35, 4/5, 0, 0, 0),
         (29/42, -2/3, 5/6, 0, 0),
         (1/6, 1/6, 5/12, 1/4, 0))
    )
    tableaux_b = np.array((11/96, 7/24, 35/96, 7/48, 1/12))
    tableaux_c = np.array((0, 2/7, 4/7, 6/7, 1))
    tableaux_d = np.array((-5/96, 1/8, -5/96, -5/48, 1/12))
    stages = len(tableaux_c)

    # guess at needed time step, and allocate space assuming that's
    # the time step.
    # TODO: More sophisticated first step choice
    dt = 0.003
    steps_guess = int(np.ceil((time_end - time_start) / dt))

    # Create arrays to store all outputs
    # The derivatives are needed to create hermite interpolants for the output
    # so we can get a good guess as to outputs even where our sim doesn't visit
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
    mgas_int = np.array([init_gas for i in range(stages)], dtype=fp)
    mstar_int = np.array([init_star for i in range(stages)], dtype=fp)
    mmetal_int = np.array([init_metal for i in range(stages)], dtype=fp)
    mdust_int = np.array([init_dust for i in range(stages)], dtype=fp)

    # package each step for ease of reading in below function calls
    # takes it from 17 lines per call to 3
    y = [mgas_int[0], mstar_int[0], mmetal_int[0], mdust_int[0]]

    # initialize the variables to hold the change in each population due to
    # different phenomena, namely star formation, inflows, outflows,
    # outflow recycling, ejecta from dying (or perhaps still living) stars,
    # as well as grain growth and destruction in the ISM and clouds
    dmgas_int = fp_zeros((stages, len(init_gas)))
    dmstar_int = fp_zeros((stages, len(init_star)))
    dmmetal_int = fp_zeros((stages, len(init_metal)))
    dmdust_int = fp_zeros((stages, len(init_dust)))

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

    sfr = sfr_model(model_params, t, redshift, *y, *interp[:-1], cache)

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

        restart_point = 0
        accepted = False

        while not accepted:
            for j in range(restart_point, stages):

                restart_point = 1
                step = tableaux_c[j] * dt

                # find out where our mini-step takes place
                mgas_int[j] = mgas_int[0] + (
                    dmgas_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                mstar_int[j] = mstar_int[0] + (
                    dmstar_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                mmetal_int[j] = mmetal_int[0] + (
                    dmmetal_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                mdust_int[j] = mdust_int[0] + (
                    dmdust_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                y = [
                    mgas_int[j],
                    mstar_int[j],
                    mmetal_int[j],
                    mdust_int[j],
                ]

                # Remake interp functions updated with derivs from the prev stage
                # and the loc of mini-step. Guess that the deriv at the current
                # mini step is equal to the deriv at the actual step.
                # TODO: This is extremely unrigorous. Figure out how to improve.
                # Maybe check the difference between the guess and what we calc,
                # and do a secant correction or something if that diff is too high,
                # Otherwise look at diff intep schema or perhaps even to a diff int
                # method.
                try:
                    gas_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((gas_masses[: i + 1], mgas_int[j])),
                        np.vstack((dgas_masses[: i + 1], dgas_masses[i])),
                    )
                    star_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((star_masses[: i + 1], mstar_int[j])),
                        np.vstack((dstar_masses[: i + 1], dstar_masses[i])),
                    )
                    metal_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((metal_masses[: i + 1], mmetal_int[j])),
                        np.vstack((dmetal_masses[: i + 1], dmetal_masses[i])),
                    )
                    dust_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((dust_masses[: i + 1], mdust_int[j])),
                        np.vstack((ddust_masses[: i + 1], ddust_masses[i])),
                    )

                    redshift = z_at_t(t + step)

                    sfr = sfr_model(
                        model_params, t + step, redshift, *
                        y, *interp[:-1], cache
                    )

                    sfr_hist = CubicSpline(
                        np.append(times[: i + 1], t + step),
                        np.append(star_formation_rates[: i + 1], sfr),
                    )

                    interp = [gas_hist, star_hist,
                              metal_hist, dust_hist, sfr_hist]

                except ValueError:
                    pass

                # calculate all derivatives at current time
                dmgas_astration = sfr * (mgas_int[j] / mgas_int[j, 0])
                dmmetal_astration = sfr * (mmetal_int[j] / mgas_int[j, 0])
                dmdust_astration = sfr * (mdust_int[j] / mgas_int[j, 0])

                dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmdust_grain_growth = grain_growth_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmdust_destruction = destruction_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_int[j] = (
                    -dmgas_astration
                    + dmgas_inflows
                    - dmgas_outflows
                    + dmgas_recycling
                    + dmgas_ejecta
                )
                dmstar_int[j] = sfr - dmgas_ejecta
                dmmetal_int[j] = (
                    -dmmetal_astration
                    + dmmetal_inflows
                    - dmmetal_outflows
                    + dmmetal_recycling
                    + dmmetal_ejecta
                )
                dmdust_int[j] = (
                    -dmdust_astration
                    + dmdust_inflows
                    - dmdust_outflows
                    + dmdust_recycling
                    + dmdust_ejecta
                    + dmdust_grain_growth
                    - dmdust_destruction
                )

                if j == 0:
                    # these derivs won't change, so store them in the output
                    dgas_masses[i] = dmgas_int[j]
                    dstar_masses[i] = dmstar_int[j]
                    dmetal_masses[i] = dmmetal_int[j]
                    ddust_masses[i] = dmdust_int[j]
                    star_formation_rates[i] = sfr

            # Using a linear comb of derivs at the current time and
            # our mini step, get both a prediction for the next timestep
            # and an estimate for the error on this timestep.
            mgas_fin = mgas_int[0] + (dmgas_int.transpose() * tableaux_b * dt).sum(
                axis=1
            )
            mstar_fin = mstar_int[0] + (dmstar_int.transpose() * tableaux_b * dt).sum(
                axis=1
            )
            mmetal_fin = mmetal_int[0] + (
                dmmetal_int.transpose() * tableaux_b * dt
            ).sum(axis=1)
            mdust_fin = mdust_int[0] + (dmdust_int.transpose() * tableaux_b * dt).sum(
                axis=1
            )

            err_gas = abs(
                (dmgas_int.transpose() * tableaux_d * dt).sum(axis=1))
            err_star = abs((dmstar_int.transpose() *
                           tableaux_d * dt).sum(axis=1))
            err_metal = abs((dmmetal_int.transpose() *
                            tableaux_d * dt).sum(axis=1))
            err_dust = abs((dmdust_int.transpose() *
                           tableaux_d * dt).sum(axis=1))

            # Condense the errors into one RMS error value, weighted inversely
            # by the tolerance in that component (abs + rel)
            err = np.hstack((err_gas, err_star, err_metal, err_dust)) / (
                absolute_tolerance
                + relative_tolerance
                * np.maximum(
                    np.hstack((mgas_fin, mstar_fin, mmetal_fin, mdust_fin)),
                    np.hstack((mgas_int[0], mstar_int[0],
                              mmetal_int[0], mdust_int[0])),
                )
            )

            err = np.sqrt((err**2).mean())

            times[i + 1] = t + dt

            cache["attempted_steps"] += 1

            # whether the step is accepted or not, we modify the step size
            # in the hopes of 0 rejections and in keeping error at 90%
            # of our tolerance.
            dt = dt * max(
                max_decrease,
                min(
                    max_increase,
                    safety_factor * (np.sqrt(1 / err) **
                                     (1 / (err_order + 1))),
                ),
            )

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
                accepted = True

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
        mgas_int[0] = mgas_fin
        mstar_int[0] = mstar_fin
        mmetal_int[0] = mmetal_fin
        mdust_int[0] = mdust_fin
        y = [mgas_int[0], mstar_int[0], mmetal_int[0], mdust_int[0]]

        gas_masses[i + 1] = mgas_int[0]
        star_masses[i + 1] = mstar_int[0]
        metal_masses[i + 1] = mmetal_int[0]
        dust_masses[i + 1] = mdust_int[0]

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

        sfr = sfr_model(model_params, t, redshift, *y, *interp[:-1], cache)

        star_formation_rates[i] = sfr

        sfr_hist = CubicSpline(
            times[: i + 1],
            star_formation_rates[: i + 1],
        )

        interp = [gas_hist, star_hist, metal_hist, dust_hist, sfr_hist]

        # Nice little progress bar
        update_progress(t / time_end)

    # calculate the derivatives at the very end, so that we can Hermite int.
    dmgas_astration = sfr * (mgas_int[0] / mgas_int[0, 0])
    dmmetal_astration = sfr * (mmetal_int[0] / mgas_int[0, 0])
    dmdust_astration = sfr * (mdust_int[0] / mgas_int[0, 0])

    dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmdust_grain_growth = grain_growth_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmdust_destruction = destruction_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
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


def evolve_5o(
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

    err_order = 4  # order of error scaling of the lower order method
    safety_factor = 0.9  # will try to keep error at this % of tolerance
    max_increase = 2  # stepsize increases by at most this factor at one time
    max_decrease = 0.5  # stepsize decreases by at most this factor at one time

    # Components of the butcher tableaux for the method, in short
    # each row of tableuax_a defines a stage of the method, and what
    # combination of derivs from each stage to use for that stage
    # must be strictly lower triangular
    # each component of tableaux_c tells what multiple of the stepsize to use
    # for the time of that stage
    # each component of tableaux_b defines the weight with which the deriv from
    # that stage contributes to the final answer to be propagated
    # each component of tableaux_d defines the weight with which the deriv from
    # that stage contributes to error estimate
    tableaux_a = np.array(
        ((0, 0, 0, 0, 0, 0, 0),
         (1/4, 0, 0, 0, 0, 0, 0),
         (1/8, 1/8, 0, 0, 0, 0, 0),
         (0, -1/2, 1, 0, 0, 0, 0),
         (13/200, -299/1000, 78/125, 13/50, 0, 0, 0),
         (548/7475, 688/2875, 572/2875, -88/575, 132/299, 0, 0),
         (37/312, 0, 4/33, 8/9, -100/117, 575/792, 0))
    )
    tableaux_b = np.array((41/520, 0, 58/165, 16/135, 50/351, 575/2376, 1/15))
    tableaux_c = np.array((0, 1/4, 1/4, 1/2, 13/20, 4/5, 1))
    tableaux_d = np.array((-31/780, 0, 38/165, -104/135, 350/351, -575/1188, 1/15))
    stages = len(tableaux_c)

    # guess at needed time step, and allocate space assuming that's
    # the time step.
    # TODO: More sophisticated first step choice
    dt = 0.003
    steps_guess = int(np.ceil((time_end - time_start) / dt))

    # Create arrays to store all outputs
    # The derivatives are needed to create hermite interpolants for the output
    # so we can get a good guess as to outputs even where our sim doesn't visit
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
    mgas_int = np.array([init_gas for i in range(stages)], dtype=fp)
    mstar_int = np.array([init_star for i in range(stages)], dtype=fp)
    mmetal_int = np.array([init_metal for i in range(stages)], dtype=fp)
    mdust_int = np.array([init_dust for i in range(stages)], dtype=fp)

    # package each step for ease of reading in below function calls
    # takes it from 17 lines per call to 3
    y = [mgas_int[0], mstar_int[0], mmetal_int[0], mdust_int[0]]

    # initialize the variables to hold the change in each population due to
    # different phenomena, namely star formation, inflows, outflows,
    # outflow recycling, ejecta from dying (or perhaps still living) stars,
    # as well as grain growth and destruction in the ISM and clouds
    dmgas_int = fp_zeros((stages, len(init_gas)))
    dmstar_int = fp_zeros((stages, len(init_star)))
    dmmetal_int = fp_zeros((stages, len(init_metal)))
    dmdust_int = fp_zeros((stages, len(init_dust)))

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

    sfr = sfr_model(model_params, t, redshift, *y, *interp[:-1], cache)

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

        restart_point = 0
        accepted = False

        while not accepted:
            for j in range(restart_point, stages):

                restart_point = 1
                step = tableaux_c[j] * dt

                # find out where our mini-step takes place
                mgas_int[j] = mgas_int[0] + (
                    dmgas_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                mstar_int[j] = mstar_int[0] + (
                    dmstar_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                mmetal_int[j] = mmetal_int[0] + (
                    dmmetal_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                mdust_int[j] = mdust_int[0] + (
                    dmdust_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                y = [
                    mgas_int[j],
                    mstar_int[j],
                    mmetal_int[j],
                    mdust_int[j],
                ]

                # Remake interp functions updated with derivs from the prev stage
                # and the loc of mini-step. Guess that the deriv at the current
                # mini step is equal to the deriv at the actual step.
                # TODO: This is extremely unrigorous. Figure out how to improve.
                # Maybe check the difference between the guess and what we calc,
                # and do a secant correction or something if that diff is too high,
                # Otherwise look at diff intep schema or perhaps even to a diff int
                # method.
                try:
                    gas_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((gas_masses[: i + 1], mgas_int[j])),
                        np.vstack((dgas_masses[: i + 1], dgas_masses[i])),
                    )
                    star_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((star_masses[: i + 1], mstar_int[j])),
                        np.vstack((dstar_masses[: i + 1], dstar_masses[i])),
                    )
                    metal_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((metal_masses[: i + 1], mmetal_int[j])),
                        np.vstack((dmetal_masses[: i + 1], dmetal_masses[i])),
                    )
                    dust_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((dust_masses[: i + 1], mdust_int[j])),
                        np.vstack((ddust_masses[: i + 1], ddust_masses[i])),
                    )

                    redshift = z_at_t(t + step)

                    sfr = sfr_model(
                        model_params, t + step, redshift, *
                        y, *interp[:-1], cache
                    )

                    sfr_hist = CubicSpline(
                        np.append(times[: i + 1], t + step),
                        np.append(star_formation_rates[: i + 1], sfr),
                    )

                    interp = [gas_hist, star_hist,
                              metal_hist, dust_hist, sfr_hist]

                except ValueError:
                    pass

                # calculate all derivatives at current time
                dmgas_astration = sfr * (mgas_int[j] / mgas_int[j, 0])
                dmmetal_astration = sfr * (mmetal_int[j] / mgas_int[j, 0])
                dmdust_astration = sfr * (mdust_int[j] / mgas_int[j, 0])

                dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmdust_grain_growth = grain_growth_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmdust_destruction = destruction_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_int[j] = (
                    -dmgas_astration
                    + dmgas_inflows
                    - dmgas_outflows
                    + dmgas_recycling
                    + dmgas_ejecta
                )
                dmstar_int[j] = sfr - dmgas_ejecta
                dmmetal_int[j] = (
                    -dmmetal_astration
                    + dmmetal_inflows
                    - dmmetal_outflows
                    + dmmetal_recycling
                    + dmmetal_ejecta
                )
                dmdust_int[j] = (
                    -dmdust_astration
                    + dmdust_inflows
                    - dmdust_outflows
                    + dmdust_recycling
                    + dmdust_ejecta
                    + dmdust_grain_growth
                    - dmdust_destruction
                )

                if j == 0:
                    # these derivs won't change, so store them in the output
                    dgas_masses[i] = dmgas_int[j]
                    dstar_masses[i] = dmstar_int[j]
                    dmetal_masses[i] = dmmetal_int[j]
                    ddust_masses[i] = dmdust_int[j]
                    star_formation_rates[i] = sfr

            # Using a linear comb of derivs at the current time and
            # our mini step, get both a prediction for the next timestep
            # and an estimate for the error on this timestep.
            mgas_fin = mgas_int[0] + (dmgas_int.transpose() * tableaux_b * dt).sum(
                axis=1
            )
            mstar_fin = mstar_int[0] + (dmstar_int.transpose() * tableaux_b * dt).sum(
                axis=1
            )
            mmetal_fin = mmetal_int[0] + (
                dmmetal_int.transpose() * tableaux_b * dt
            ).sum(axis=1)
            mdust_fin = mdust_int[0] + (dmdust_int.transpose() * tableaux_b * dt).sum(
                axis=1
            )

            err_gas = abs(
                (dmgas_int.transpose() * tableaux_d * dt).sum(axis=1))
            err_star = abs((dmstar_int.transpose() *
                           tableaux_d * dt).sum(axis=1))
            err_metal = abs((dmmetal_int.transpose() *
                            tableaux_d * dt).sum(axis=1))
            err_dust = abs((dmdust_int.transpose() *
                           tableaux_d * dt).sum(axis=1))

            # Condense the errors into one RMS error value, weighted inversely
            # by the tolerance in that component (abs + rel)
            err = np.hstack((err_gas, err_star, err_metal, err_dust)) / (
                absolute_tolerance
                + relative_tolerance
                * np.maximum(
                    np.hstack((mgas_fin, mstar_fin, mmetal_fin, mdust_fin)),
                    np.hstack((mgas_int[0], mstar_int[0],
                              mmetal_int[0], mdust_int[0])),
                )
            )

            err = np.sqrt((err**2).mean())

            times[i + 1] = t + dt

            cache["attempted_steps"] += 1

            # whether the step is accepted or not, we modify the step size
            # in the hopes of 0 rejections and in keeping error at 90%
            # of our tolerance.
            dt = dt * max(
                max_decrease,
                min(
                    max_increase,
                    safety_factor * (np.sqrt(1 / err) **
                                     (1 / (err_order + 1))),
                ),
            )

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
                accepted = True

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
        mgas_int[0] = mgas_fin
        mstar_int[0] = mstar_fin
        mmetal_int[0] = mmetal_fin
        mdust_int[0] = mdust_fin
        y = [mgas_int[0], mstar_int[0], mmetal_int[0], mdust_int[0]]

        gas_masses[i + 1] = mgas_int[0]
        star_masses[i + 1] = mstar_int[0]
        metal_masses[i + 1] = mmetal_int[0]
        dust_masses[i + 1] = mdust_int[0]

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

        sfr = sfr_model(model_params, t, redshift, *y, *interp[:-1], cache)

        star_formation_rates[i] = sfr

        sfr_hist = CubicSpline(
            times[: i + 1],
            star_formation_rates[: i + 1],
        )

        interp = [gas_hist, star_hist, metal_hist, dust_hist, sfr_hist]

        # Nice little progress bar
        update_progress(t / time_end)

    # calculate the derivatives at the very end, so that we can Hermite int.
    dmgas_astration = sfr * (mgas_int[0] / mgas_int[0, 0])
    dmmetal_astration = sfr * (mmetal_int[0] / mgas_int[0, 0])
    dmdust_astration = sfr * (mdust_int[0] / mgas_int[0, 0])

    dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmdust_grain_growth = grain_growth_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmdust_destruction = destruction_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
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


def evolve_6o(
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

    err_order = 5  # order of error scaling of the lower order method
    safety_factor = 0.9  # will try to keep error at this % of tolerance
    max_increase = 2  # stepsize increases by at most this factor at one time
    max_decrease = 0.5  # stepsize decreases by at most this factor at one time

    # Components of the butcher tableaux for the method, in short
    # each row of tableuax_a defines a stage of the method, and what
    # combination of derivs from each stage to use for that stage
    # must be strictly lower triangular
    # each component of tableaux_c tells what multiple of the stepsize to use
    # for the time of that stage
    # each component of tableaux_b defines the weight with which the deriv from
    # that stage contributes to the final answer to be propagated
    # each component of tableaux_d defines the weight with which the deriv from
    # that stage contributes to error estimate
    tableaux_a = np.array(
        ((0, 0, 0, 0, 0, 0, 0, 0),
         (1/18, 0, 0, 0, 0, 0, 0, 0),
         (-1/12, 1/4, 0, 0, 0, 0, 0, 0),
         (-2/81, 4/27, 8/81, 0, 0, 0, 0, 0),
         (40/33, -4/11, -56/11, 54/11, 0, 0, 0, 0),
         (-369/73, 72/73, 5380/219, -12285/584, 2695/1752, 0, 0, 0),
         (-8716/891, 656/297, 39520/891, -416/11, 52/27, 0, 0, 0),
         (3015/256, -9/4, -4219/78, 5985/128, -539/384, 0, 693/3328, 0))
    )
    tableaux_b = np.array((57/640, 0, -16/65, 1377/2240, 121/320, 0, 891/8320, 2/35))
    tableaux_c = np.array((0, 1/18, 1/6, 2/9, 2/3, 1, 8/9, 1))
    tableaux_d = np.array((33/640, 0, -132/325, 891/2240, -33/320, -73/700, 891/8320, 2/35))
    stages = len(tableaux_c)

    # guess at needed time step, and allocate space assuming that's
    # the time step.
    # TODO: More sophisticated first step choice
    dt = 0.003
    steps_guess = int(np.ceil((time_end - time_start) / dt))

    # Create arrays to store all outputs
    # The derivatives are needed to create hermite interpolants for the output
    # so we can get a good guess as to outputs even where our sim doesn't visit
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
    mgas_int = np.array([init_gas for i in range(stages)], dtype=fp)
    mstar_int = np.array([init_star for i in range(stages)], dtype=fp)
    mmetal_int = np.array([init_metal for i in range(stages)], dtype=fp)
    mdust_int = np.array([init_dust for i in range(stages)], dtype=fp)

    # package each step for ease of reading in below function calls
    # takes it from 17 lines per call to 3
    y = [mgas_int[0], mstar_int[0], mmetal_int[0], mdust_int[0]]

    # initialize the variables to hold the change in each population due to
    # different phenomena, namely star formation, inflows, outflows,
    # outflow recycling, ejecta from dying (or perhaps still living) stars,
    # as well as grain growth and destruction in the ISM and clouds
    dmgas_int = fp_zeros((stages, len(init_gas)))
    dmstar_int = fp_zeros((stages, len(init_star)))
    dmmetal_int = fp_zeros((stages, len(init_metal)))
    dmdust_int = fp_zeros((stages, len(init_dust)))

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

    sfr = sfr_model(model_params, t, redshift, *y, *interp[:-1], cache)

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

        restart_point = 0
        accepted = False

        while not accepted:
            for j in range(restart_point, stages):

                restart_point = 1
                step = tableaux_c[j] * dt

                # find out where our mini-step takes place
                mgas_int[j] = mgas_int[0] + (
                    dmgas_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                mstar_int[j] = mstar_int[0] + (
                    dmstar_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                mmetal_int[j] = mmetal_int[0] + (
                    dmmetal_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                mdust_int[j] = mdust_int[0] + (
                    dmdust_int.transpose() * tableaux_a[j] * dt
                ).sum(axis=1)
                y = [
                    mgas_int[j],
                    mstar_int[j],
                    mmetal_int[j],
                    mdust_int[j],
                ]

                # Remake interp functions updated with derivs from the prev stage
                # and the loc of mini-step. Guess that the deriv at the current
                # mini step is equal to the deriv at the actual step.
                # TODO: This is extremely unrigorous. Figure out how to improve.
                # Maybe check the difference between the guess and what we calc,
                # and do a secant correction or something if that diff is too high,
                # Otherwise look at diff intep schema or perhaps even to a diff int
                # method.
                try:
                    gas_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((gas_masses[: i + 1], mgas_int[j])),
                        np.vstack((dgas_masses[: i + 1], dgas_masses[i])),
                    )
                    star_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((star_masses[: i + 1], mstar_int[j])),
                        np.vstack((dstar_masses[: i + 1], dstar_masses[i])),
                    )
                    metal_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((metal_masses[: i + 1], mmetal_int[j])),
                        np.vstack((dmetal_masses[: i + 1], dmetal_masses[i])),
                    )
                    dust_hist = CubicHermiteSpline(
                        np.append(times[: i + 1], t + step),
                        np.vstack((dust_masses[: i + 1], mdust_int[j])),
                        np.vstack((ddust_masses[: i + 1], ddust_masses[i])),
                    )

                    redshift = z_at_t(t + step)

                    sfr = sfr_model(
                        model_params, t + step, redshift, *
                        y, *interp[:-1], cache
                    )

                    sfr_hist = CubicSpline(
                        np.append(times[: i + 1], t + step),
                        np.append(star_formation_rates[: i + 1], sfr),
                    )

                    interp = [gas_hist, star_hist,
                              metal_hist, dust_hist, sfr_hist]

                except ValueError:
                    pass

                # calculate all derivatives at current time
                dmgas_astration = sfr * (mgas_int[j] / mgas_int[j, 0])
                dmmetal_astration = sfr * (mmetal_int[j] / mgas_int[j, 0])
                dmdust_astration = sfr * (mdust_int[j] / mgas_int[j, 0])

                dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmdust_grain_growth = grain_growth_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmdust_destruction = destruction_model(
                    model_params, sfr, imf, t + step, redshift, *y, *interp, cache
                )

                dmgas_int[j] = (
                    -dmgas_astration
                    + dmgas_inflows
                    - dmgas_outflows
                    + dmgas_recycling
                    + dmgas_ejecta
                )
                dmstar_int[j] = sfr - dmgas_ejecta
                dmmetal_int[j] = (
                    -dmmetal_astration
                    + dmmetal_inflows
                    - dmmetal_outflows
                    + dmmetal_recycling
                    + dmmetal_ejecta
                )
                dmdust_int[j] = (
                    -dmdust_astration
                    + dmdust_inflows
                    - dmdust_outflows
                    + dmdust_recycling
                    + dmdust_ejecta
                    + dmdust_grain_growth
                    - dmdust_destruction
                )

                if j == 0:
                    # these derivs won't change, so store them in the output
                    dgas_masses[i] = dmgas_int[j]
                    dstar_masses[i] = dmstar_int[j]
                    dmetal_masses[i] = dmmetal_int[j]
                    ddust_masses[i] = dmdust_int[j]
                    star_formation_rates[i] = sfr

            # Using a linear comb of derivs at the current time and
            # our mini step, get both a prediction for the next timestep
            # and an estimate for the error on this timestep.
            mgas_fin = mgas_int[0] + (dmgas_int.transpose() * tableaux_b * dt).sum(
                axis=1
            )
            mstar_fin = mstar_int[0] + (dmstar_int.transpose() * tableaux_b * dt).sum(
                axis=1
            )
            mmetal_fin = mmetal_int[0] + (
                dmmetal_int.transpose() * tableaux_b * dt
            ).sum(axis=1)
            mdust_fin = mdust_int[0] + (dmdust_int.transpose() * tableaux_b * dt).sum(
                axis=1
            )

            err_gas = abs(
                (dmgas_int.transpose() * tableaux_d * dt).sum(axis=1))
            err_star = abs((dmstar_int.transpose() *
                           tableaux_d * dt).sum(axis=1))
            err_metal = abs((dmmetal_int.transpose() *
                            tableaux_d * dt).sum(axis=1))
            err_dust = abs((dmdust_int.transpose() *
                           tableaux_d * dt).sum(axis=1))

            # Condense the errors into one RMS error value, weighted inversely
            # by the tolerance in that component (abs + rel)
            err = np.hstack((err_gas, err_star, err_metal, err_dust)) / (
                absolute_tolerance
                + relative_tolerance
                * np.maximum(
                    np.hstack((mgas_fin, mstar_fin, mmetal_fin, mdust_fin)),
                    np.hstack((mgas_int[0], mstar_int[0],
                              mmetal_int[0], mdust_int[0])),
                )
            )

            err = np.sqrt((err**2).mean())

            times[i + 1] = t + dt

            cache["attempted_steps"] += 1

            # whether the step is accepted or not, we modify the step size
            # in the hopes of 0 rejections and in keeping error at 90%
            # of our tolerance.
            dt = dt * max(
                max_decrease,
                min(
                    max_increase,
                    safety_factor * (np.sqrt(1 / err) **
                                     (1 / (err_order + 1))),
                ),
            )

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
                accepted = True

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
        mgas_int[0] = mgas_fin
        mstar_int[0] = mstar_fin
        mmetal_int[0] = mmetal_fin
        mdust_int[0] = mdust_fin
        y = [mgas_int[0], mstar_int[0], mmetal_int[0], mdust_int[0]]

        gas_masses[i + 1] = mgas_int[0]
        star_masses[i + 1] = mstar_int[0]
        metal_masses[i + 1] = mmetal_int[0]
        dust_masses[i + 1] = mdust_int[0]

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

        sfr = sfr_model(model_params, t, redshift, *y, *interp[:-1], cache)

        star_formation_rates[i] = sfr

        sfr_hist = CubicSpline(
            times[: i + 1],
            star_formation_rates[: i + 1],
        )

        interp = [gas_hist, star_hist, metal_hist, dust_hist, sfr_hist]

        # Nice little progress bar
        update_progress(t / time_end)

    # calculate the derivatives at the very end, so that we can Hermite int.
    dmgas_astration = sfr * (mgas_int[0] / mgas_int[0, 0])
    dmmetal_astration = sfr * (mmetal_int[0] / mgas_int[0, 0])
    dmdust_astration = sfr * (mdust_int[0] / mgas_int[0, 0])

    dmgas_inflows, dmmetal_inflows, dmdust_inflows = inflow_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmgas_outflows, dmmetal_outflows, dmdust_outflows = outflow_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmgas_recycling, dmmetal_recycling, dmdust_recycling = recycling_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmgas_ejecta, dmmetal_ejecta, dmdust_ejecta = ejecta_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmdust_grain_growth = grain_growth_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
    )

    dmdust_destruction = destruction_model(
        model_params, sfr, imf, t, redshift, *y, *interp, cache
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
    if not isinstance(progress, fp):
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
