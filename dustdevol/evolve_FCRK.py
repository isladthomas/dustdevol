from numpy import (
    finfo,
    ceil,
    inf,
    tensordot,
    append,
    hstack,
    vstack,
    maximum,
    sqrt,
)
from dustdevol.generic import fp, fp_zeros, fp_empty, fp_array, fp_full, z_at_t
from scipy.interpolate import CubicSpline, PPoly
from sys import stdout


def evolve_2o_FC(
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
    """Abstract Method for evolving a chem-evol model using user-specified models for
    inflows, outflows, recycling, grain growth, dust destruction, and stellar ejecta.
    Uses 2nd order Functional Continuous Runge-Kutta with an embedded 1st order FCRK
    for error estimation and step-size selection, both methods taken from Tavernini 71

    Parameters
    ----------
    time_start : float
                 Starting time for the simulation.
    time_end : float
               End time for the simulation.
    sfr_model : function(dict, float, float, 4*ndarray, 4*function, dict) -> (s,)
                Model which outputs the sfr when passed model_params, current time,
                redshift, current gas, star, metal and dust masses, function to get
                previous gas, star, metal and dust masses, and the model cache.
    imf : function(float) -> float
          Function which gives the proportion of stars formed in the mass interval
          m + dm.
    inflow_model : function(dict, float, float, 4*ndarray, 5*function, dict) -> ((g,),(m,),(d,))
                   Model which outputs the amount of gas, metals, and dust gained from inflows
                   when given all the parameters of sfr_model, as well as a function giving
                   previous sfr.
    outflow_model : function(dict, float, float, 4*ndarray, 5*function, dict) -> ((g,),(m,),(d,))
                    Model which outputs the amount of gas, metals, and dust lost to outflows
                    when given all parameters of inflow_model.
    recycling_model : function(dict, float, float, 4*ndarray, 5*function, dict) -> ((g,),(m,),(d,))
                      Model which gives the amount of gas, metals, and dust regained due to
                      outflow recycling.
    grain_growth_model : function(dict, float, float, 4*ndarray, 5*function, dict) -> (d,)
                         Model which gives the amount of dust gained due to grain growth.
    dust_destruction_model : function(dict, float, float, 4*ndarray, 5*function, dict) -> (d,)
                             Model which gives the amount of dust destroyed via processes like
                             SNe shockwaves and photofragmentation.
    ejecta_model : function(dict, float, float, 4*ndarray, 5*function, dict) -> ((g,),(m,),(d,))
                   Model which gives the amount of gas, metals, and dust produced
                   from dying stars.
    init_gas : (g,)
               ndarray representing the various gas populations. Interpretation depends on
               Models used.
    init_star : (s,)
                ndarray representing the various stellar populations. Interpretation depends
                on Models used.
    init_metal : (m,)
                 ndarray representing the various metal populations. Interpretation depends
                 on Models used.
    init_dust : (d,)
                ndarray representing the various dust populations. Interpretation depends
                on Models used.
    model_params : dict
                   Dictionary containing assorted parameters for the various models.
    absolute_tolerance : float
                         Constant term used to scale error. Roughly, the error tolerance will
                         be at least this much, regardless of how small the values are.
                         Must be nonzero, regardless of relative_tolerance, or else
                         error will be NaN if any component is zero.
    relative_tolerance : float
                         Term used to scale error that scales with the size of each component.
                         multiplied by the maximum of the current and proposed values.

    Returns
    -------
    out : dict
          dictionary providing all results, including times visited,
          calculated values and derivatives for gas, stars, metals, dust,
          and sfr (no derivatives), interpolants for these values, as well
          as the cache.
    """

    # how small we can let the step size get before raising an error
    # needs to be set because if step size is too small, t + dt will be
    # identical to t due to limited precision
    # this will raise an error anyway, but might as well give more info
    # as to why
    eps = 100 * finfo(fp).eps

    err_order = fp(1)  # order of error scaling of the lower order method
    safety_factor = fp(0.9)  # will try to keep error at this % of tolerance
    # stepsize increases by at most this factor in one timestep
    max_increase = fp(2)
    # stepsize decreases by at most this factor in one timestep
    max_decrease = fp(0.5)

    # Components of the butcher tableaux for the method, in short
    # each row of tableuax_a defines a stage of the method, and what
    # combination of derivs from each stage to use for that stage
    # expanded to polynomials in the time along the step, normalized to
    # [0,1], to allow interpolation at each stage
    # must be strictly lower triangular
    # each component of tableaux_c tells what multiple of the stepsize to use
    # for the time of that stage
    # each component of tableaux_b defines the weight with which the deriv from
    # that stage contributes to the final answer to be propagated
    # again expanded to polynomials in (Functional) Continuous Runge-Kutta
    # each component of tableaux_d defines the weight with which the deriv from
    # that stage contributes to error estimate
    tableaux_a = fp_array((((0, 0, 0), (0, 0, 0)), ((0, 1, 0), (0, 0, 0))))
    tableaux_b = fp_array(((-1 / 2, 1, 0), (1 / 2, 0, 0)))
    tableaux_c = fp_array((0, 1))
    tableaux_d = fp_array((-1 / 2, 1 / 2))
    stages = len(tableaux_c)
    interp_order = tableaux_a.shape[2]

    # guess at needed time step, and allocate space assuming that's
    # the time step.
    # TODO: More sophisticated first step choice
    dt = fp(0.003)
    steps_guess = int(ceil((time_end - time_start) / dt))

    # Create arrays to store all outputs, including derivatives
    times = fp_empty(steps_guess)
    times[:] = inf
    gas_masses = fp_empty((steps_guess, len(init_gas)))
    star_masses = fp_empty((steps_guess, len(init_star)))
    metal_masses = fp_empty((steps_guess, len(init_metal)))
    dust_masses = fp_empty((steps_guess, len(init_dust)))

    times[0] = fp_array(time_start)
    gas_masses[0] = fp_array(init_gas)
    star_masses[0] = fp_array(init_star)
    metal_masses[0] = fp_array(init_metal)
    dust_masses[0] = fp_array(init_dust)

    dgas_masses = fp_empty((steps_guess, len(init_gas)))
    dstar_masses = fp_empty((steps_guess, len(init_star)))
    dmetal_masses = fp_empty((steps_guess, len(init_metal)))
    ddust_masses = fp_empty((steps_guess, len(init_dust)))

    star_formation_rates = fp_empty(steps_guess)

    # initialize the variables to actually be evolved during the sim loop
    t = fp(time_start)
    mgas_int = fp_array([init_gas for i in range(stages)])
    mstar_int = fp_array([init_star for i in range(stages)])
    mmetal_int = fp_array([init_metal for i in range(stages)])
    mdust_int = fp_array([init_dust for i in range(stages)])

    # package each step for ease of reading in below function calls
    # takes it from 17 lines per call to 3
    y = [mgas_int[0], mstar_int[0], mmetal_int[0], mdust_int[0]]

    # initialize the variables to hold the change in each population due to
    # different phenomena, namely star formation, inflows, outflows,
    # outflow recycling, ejecta from dying (or perhaps still living) stars,
    # as well as grain growth and destruction in the ISM and clouds
    dmgas_int = fp_empty((stages, len(init_gas)))
    dmstar_int = fp_empty((stages, len(init_star)))
    dmmetal_int = fp_empty((stages, len(init_metal)))
    dmdust_int = fp_empty((stages, len(init_dust)))

    dmgas_astration = fp_empty(len(init_gas))
    dmmetal_astration = fp_empty(len(init_metal))
    dmdust_astration = fp_empty(len(init_dust))

    dmgas_inflows = fp_empty(len(init_gas))
    dmmetal_inflows = fp_empty(len(init_metal))
    dmdust_inflows = fp_empty(len(init_dust))

    dmgas_outflows = fp_empty(len(init_gas))
    dmmetal_outflows = fp_empty(len(init_metal))
    dmdust_outflows = fp_empty(len(init_dust))

    dmgas_recycling = fp_empty((len(init_gas)))
    dmmetal_recycling = fp_empty((len(init_metal)))
    dmdust_recycling = fp_empty((len(init_dust)))

    dmgas_ejecta = fp_empty(len(init_gas))
    dmmetal_ejecta = fp_empty(len(init_metal))
    dmdust_ejecta = fp_empty(len(init_dust))

    dmdust_grain_growth = fp_empty(len(init_dust))
    dmdust_destruction = fp_empty(len(init_dust))

    # Keep track of what step we're on, and also provide
    # a dictionary that lets functions give additional output/
    # store certain results in memory to speed up later computation.
    i = 0
    cache = {}

    # Initialize interpolants for the history of our galaxy
    # assuming all masses held constant for 1 Gyr before time_start
    # exact values for this part don't really matter, ideally these values
    # aren't even used, but the polynomial must be constructed for the main
    # loop to work.
    coeffs = fp_zeros((interp_order, 1, len(init_gas)))
    coeffs[-1, :, :] = init_gas
    gas_hist = PPoly(coeffs, [time_start - 1, time_start])

    coeffs = fp_zeros((interp_order, 1, len(init_star)))
    coeffs[-1, :, :] = init_star
    star_hist = PPoly(coeffs, [time_start - 1, time_start])

    coeffs = fp_zeros((interp_order, 1, len(init_metal)))
    coeffs[-1, :, :] = init_metal
    metal_hist = PPoly(coeffs, [time_start - 1, time_start])

    coeffs = fp_zeros((interp_order, 1, len(init_dust)))
    coeffs[-1, :, :] = init_dust
    dust_hist = PPoly(coeffs, [time_start - 1, time_start])

    redshift = z_at_t(t)

    interp = [gas_hist, star_hist, metal_hist, dust_hist, None]

    # We don't have good derivative estimates for the sfr so it is
    # constructed through Cubic Hermite Spline interpolation
    # however contructing a spline who's values won't be used is wasteful
    # so we just make a quick function that returns the starting sfr
    sfr = sfr_model(model_params, t, redshift, *y, *interp[:-1], cache)

    star_formation_rates[0] = sfr

    def sfr_hist(t):
        if hasattr(t, "shape"):
            r = fp_empty(t.shape)
            r[:] = sfr
            return r
        else:
            return fp_array(sfr)

    interp[-1] = sfr_hist

    # Keep track of how many steps were attempted. If a large number
    # of steps fail, that could mean that the problem is stiff
    # or that there are breaking points causing problems.
    # Switch to an implicit method in this case.
    cache["attempted_steps"] = 0

    while t < time_end:

        # stage 0 only needs to be computed once, so if a step is rejected,
        # restart at stage 1
        restart_point = 0
        accepted = False

        while not accepted:

            # dividing factor which normalizes the polynomial argument to be
            # [0,1] on the interval [t, t+dt], divided by dt
            theta_stretch = fp_array(
                [[dt ** (interp_order - k - 2)]
                 for k in range(interp_order - 1)]
            )

            for j in range(restart_point, stages):

                restart_point = 1
                step = tableaux_c[j] * dt

                # Remake interp funcs using the FCRK prescription from tableaux
                if j == 0:
                    coeffs = fp_zeros((3, 1, len(mgas_int[j])))
                    coeffs[-1, :, :] = mgas_int[j]
                    gas_hist.extend(coeffs, [t + dt])

                    coeffs = fp_zeros((3, 1, len(mstar_int[j])))
                    coeffs[-1, :, :] = mstar_int[j]
                    star_hist.extend(coeffs, [t + dt])

                    coeffs = fp_zeros((3, 1, len(mmetal_int[j])))
                    coeffs[-1, :, :] = mmetal_int[j]
                    metal_hist.extend(coeffs, [t + dt])

                    coeffs = fp_zeros((3, 1, len(mdust_int[j])))
                    coeffs[-1, :, :] = mdust_int[j]
                    dust_hist.extend(coeffs, [t + dt])

                else:

                    gas_hist.c[:-1, -1] = (
                        tensordot(tableaux_a[j], dmgas_int,
                                  axes=[[0], [0]])[:-1]
                        / theta_stretch
                    )
                    star_hist.c[:-1, -1] = (
                        tensordot(tableaux_a[j], dmstar_int,
                                  axes=[[0], [0]])[:-1]
                        / theta_stretch
                    )
                    metal_hist.c[:-1, -1] = (
                        tensordot(tableaux_a[j], dmmetal_int,
                                  axes=[[0], [0]])[:-1]
                        / theta_stretch
                    )
                    dust_hist.c[:-1, -1] = (
                        tensordot(tableaux_a[j], dmdust_int,
                                  axes=[[0], [0]])[:-1]
                        / theta_stretch
                    )
                    # if a step is rejected, reset the interpolants to
                    # the new step size.
                    gas_hist.x[-1] = t + dt
                    star_hist.x[-1] = t + dt
                    metal_hist.x[-1] = t + dt
                    dust_hist.x[-1] = t + dt

                    redshift = z_at_t(t + step)

                    sfr = sfr_model(
                        model_params, t + step, redshift, *
                        y, *interp[:-1], cache
                    )

                    sfr_hist = CubicSpline(
                        append(times[: i + 1], t + step),
                        append(star_formation_rates[: i + 1], sfr),
                    )

                    interp = [gas_hist, star_hist,
                              metal_hist, dust_hist, sfr_hist]

                # find out where our mini-step takes place
                mgas_int[j] = gas_hist(t + step)
                mstar_int[j] = star_hist(t + step)
                mmetal_int[j] = metal_hist(t + step)
                mdust_int[j] = dust_hist(t + step)
                y = [
                    mgas_int[j],
                    mstar_int[j],
                    mmetal_int[j],
                    mdust_int[j],
                ]

                # calculate all derivatives at current time
                while True:
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

                    # check that outflows are not too large (50% of gas blown
                    # out in 30 Myr). If they are, rescale sfr and recalculate
                    # the gas stuff.
                    # NOTE: We include a little "fudge factor" when checking
                    # size of outflows. This is due to floating point
                    # shenanigans; the condition can fail if there's not enough prec
                    # for an almost-1 rescaling factor to actually change the sfr
                    if dmgas_outflows <= (0.5 * mgas_int[j, 0] / 0.03) * 1.001:
                        break
                    else:
                        print((0.5 * mgas_int[j, 0]) / (0.03 * dmgas_outflows))

                        sfr *= (0.5 * mgas_int[j, 0]) / (0.03 * dmgas_outflows)

                        if j == 0:

                            star_formation_rates[i] = sfr

                            sfr_hist = CubicSpline(
                                times[: i + 1],
                                star_formation_rates[: i + 1],
                            )

                            interp = [gas_hist, star_hist, metal_hist, dust_hist, sfr_hist]

                        else:

                            sfr_hist = CubicSpline(
                                append(times[: i + 1], t + step),
                                append(star_formation_rates[: i + 1], sfr),
                            )

                            interp = [gas_hist, star_hist,
                                      metal_hist, dust_hist, sfr_hist]

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
            # our mini step, get a prediction for the next timestep,
            # an interpolant to use in future timesteps,
            # and an estimate for the error on this timestep.
            gas_hist.c[:-1, -1] = (
                tensordot(tableaux_b, dmgas_int, axes=[[0], [0]])[
                    :-1] / theta_stretch
            )
            star_hist.c[:-1, -1] = (
                tensordot(tableaux_b, dmstar_int, axes=[
                          [0], [0]])[:-1] / theta_stretch
            )
            star_hist.c[:-1, -1] = (
                tensordot(tableaux_b, dmstar_int, axes=[
                          [0], [0]])[:-1] / theta_stretch
            )
            star_hist.c[:-1, -1] = (
                tensordot(tableaux_b, dmstar_int, axes=[
                          [0], [0]])[:-1] / theta_stretch
            )

            mgas_fin = gas_hist(t + dt)
            mstar_fin = star_hist(t + dt)
            mmetal_fin = metal_hist(t + dt)
            mdust_fin = dust_hist(t + dt)

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
            err = hstack((err_gas, err_star, err_metal, err_dust)) / (
                fp(absolute_tolerance)
                + fp(relative_tolerance)
                * maximum(
                    hstack((mgas_fin, mstar_fin, mmetal_fin, mdust_fin)),
                    hstack((mgas_int[0], mstar_int[0],
                           mmetal_int[0], mdust_int[0])),
                )
            )

            err = sqrt((err**2).mean())

            times[i + 1] = t + dt

            cache["attempted_steps"] += 1

            # whether the step is accepted or not, we modify the step size
            # in the hopes of 0 rejections and in keeping error at 90%
            # of our tolerance.
            dt = dt * max(
                max_decrease,
                min(
                    max_increase,
                    safety_factor * (sqrt(1 / err) ** (1 / (err_order + 1))),
                ),
            )

            # if the time step becomes extremely small, errstop
            # as we can reach a point where t + dt is identical to t
            # floating point shenaniganery :/
            if times[i + 1] - times[i] <= eps:
                raise RuntimeError(
                    "Stepsize too small for defined precision. Either increasing or decreasing tolerance can help, though increasing is more likely."
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
            times = append(times, fp_full(n, inf))
            gas_masses = vstack((gas_masses, fp_empty((n, len(init_gas)))))
            star_masses = vstack((star_masses, fp_empty((n, len(init_star)))))
            metal_masses = vstack(
                (metal_masses, fp_empty((n, len(init_metal)))))
            dust_masses = vstack((dust_masses, fp_empty((n, len(init_dust)))))

            dgas_masses = vstack((dgas_masses, fp_empty((n, len(init_gas)))))
            dstar_masses = vstack(
                (dstar_masses, fp_empty((n, len(init_star)))))
            dmetal_masses = vstack(
                (dmetal_masses, fp_empty((n, len(init_metal)))))
            ddust_masses = vstack(
                (ddust_masses, fp_empty((n, len(init_dust)))))

            star_formation_rates = append(star_formation_rates, fp_empty(n))

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
    to_keep = times != inf

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
        "gas_func": gas_hist,
        "star_func": star_hist,
        "metal_func": metal_hist,
        "dust_func": dust_hist,
        "sfr": star_formation_rates[to_keep],
        "sfr_func": sfr_hist,
        "cache": cache,
    }

    return results


def update_progress(progress):
    """
    From user Brian Khuu on stack exchange, displays a nice little
    progress bar.

    Parameters
    ----------
    progress : int or float
               Amount of progress, between 0 and 1.

    Returns
    -------
    out : None
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
    stdout.write(text)
    stdout.flush()
