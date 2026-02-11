from numpy import (
    array,
    logspace,
    where,
    diff,
    vectorize,
    log10,
    searchsorted,
    clip,
    hstack,
    maximum,
    sqrt,
)


def life_from_mass_vec(masses, stellar_lifetimes, metallicity):
    """
    helper function which uses 0-order interpolation to find the lifetime
    in Gyr given a mass in Msol
    """

    masses_half = stellar_lifetimes[:-1, 0] / 2 + stellar_lifetimes[1:, 0] / 2
    eff_indices = searchsorted(masses_half, masses)
    eff_indices = clip(eff_indices, 0, len(stellar_lifetimes[:, 0]) - 1)

    if metallicity == "high":
        return stellar_lifetimes[eff_indices, 2]
    else:
        return stellar_lifetimes[eff_indices, 1]


def remnant_mass(m):
    """
    calculates how much mass of the star remains in stellar remnants.
    """

    rem_mass = where(m < 25, 1.5, 0.61 * m - 13.75)
    rem_mass[m <= 9] = (0.106 * m + 0.446)[m <= 9]

    return rem_mass


def fresh_metals(yield_table, metallicity_cutoffs, masses, metallicity):
    """
    nab the amount of metals generated in the death of a star of mass m
    from the yield table
    """

    i = searchsorted(metallicity_cutoffs, metallicity)[0]
    stepsize = len(metallicity)

    masses_half = yield_table[:-1, 0] / 2 + yield_table[1:, 0] / 2
    eff_indices = searchsorted(masses_half, masses)
    eff_indices = clip(eff_indices, 0, len(yield_table[:, 0]) - 1)
    return yield_table[eff_indices, i * stepsize + 1: (i + 1) * stepsize + 1]


def fresh_dust(
    eff_table,
    ejected_metals,
    reduction_factor,
    masses,
):
    """
    nab the amount of dust generated in the death of a star of mass m,
    either from the yield table, if the star goes supernova,
    or from a fraction of the metals generated, if the star becomes a PN
    """

    masses_half = eff_table[:-1, 0] / 2 + eff_table[1:, 0] / 2
    eff_indices = searchsorted(masses_half, masses, side="left")
    eff_indices = clip(eff_indices, 0, len(eff_table[:, 0]) - 1)
    dust_eff = eff_table[eff_indices, 1] / reduction_factor
    dust_eff[masses <= 8] = 0.15
    dust_eff[masses > 40] = 0

    return dust_eff * ejected_metals


def fast_ejecta(
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
    calculate the gas, metals, and dust emmitted from dying stars, given
    a function for mass of stellar remnants as well as output tables for
    metal and dust yields. In essence, convolves the past SFR with the IMF
    and a yield function to find how much gas/metal/dust is beind shot out now
    requires:
        - \"dust_yields\": table where each row gives a mass in Msol, and the
                           dust *created*, not recycled, when such a star dies
        - \"metal_yields\": table where each row gives a mass in Msol, followed
                            by several entries giving the metals created when
                            such a star dies, ordered the same as init_metals,
                            repeated for each metallicity level
        - \"yield_table_z_cutoffs\": list giving the cutoff for each
                                     metallicity level in the metal_yields table
        - \"sn_dust_reduction\": factor which divides dust created in supernovae
        - \"stellar_lifetimes\": table where each row gives, in order
                                 the mass of a star in Msol, the lifetime
                                 of such a star in Gyrs in a low metallicity
                                 (Z < 0.008) environment, and the lifetime in
                                 a high metallicity (Z >= 0.008) environment
    NOTE: does modify model_params, storing an additional value with key
    \"z_history\", which allows the function to access historical metallicities
    """

    # store all model params for easier passing to subroutines
    dust_yield_table = model_params["dust_yields"]
    metal_yield_table = model_params["metal_yields"]
    metallicity_cutoffs = model_params["yield_table_z_cutoffs"]
    sn_reduction = model_params["sn_dust_reduction"]
    stellar_lifetimes = model_params["stellar_lifetimes"]

    # grab everything precomputable, and precompute it if not
    # specifically, the masses we sample, and the imfs, ejecta (m - rem)
    # and the size of the window at each mass
    try:
        masses = model_params["ejecta_masses"]
        ejecta = model_params["ejecta_vals"]
        imf_vals = model_params["imf_values"]
        d_masses = model_params["d_masses"]

    except KeyError:

        model_params["ejecta_masses"] = logspace(log10(0.8), log10(120), 513)

        # get mass windows
        masses = model_params["ejecta_masses"]
        model_params["d_masses"] = diff(masses)
        d_masses = model_params["d_masses"]

        # switch "masses" to the midpoints, instead of left edges
        model_params["ejecta_masses"] = masses[:-1] + (d_masses / 2)
        masses = model_params["ejecta_masses"]

        # calcualte imf and ejecta at midpoints
        model_params["imf_values"] = imf(masses)
        imf_vals = model_params["imf_values"]
        remnants = remnant_mass(masses)
        model_params["ejecta_vals"] = masses - remnants
        ejecta = model_params["ejecta_vals"]

    # determine if high or low metallicity lifetimes are to be used
    if (mmetal[0] / mgas[0]) <= 0.008:
        metallicity = "low"

    else:
        metallicity = "high"

    lifetimes = life_from_mass_vec(masses, stellar_lifetimes, metallicity)

    d_masses = where(t > lifetimes, d_masses, 0)

    # create arrays for historical metallicity and sfr
    z_at_birth = metal_hist(t - lifetimes) / gas_hist(t - lifetimes)
    sfr_vals = sfr_hist(t - lifetimes)

    # calculate all our ejecta
    ejected_gas = (ejecta * sfr_vals * imf_vals * d_masses).sum(axis=0)

    fresh_metal_ejecta = fresh_metals(
        metal_yield_table, metallicity_cutoffs, masses, mmetal / mgas[0]
    )
    old_metal_ejecta = ejecta[:, None] * z_at_birth
    ejected_metal = (
        (fresh_metal_ejecta + old_metal_ejecta)
        * sfr_vals[:, None]
        * imf_vals[:, None]
        * d_masses[:, None]
    ).sum(axis=0)

    fresh_dust_ejecta = fresh_dust(
        dust_yield_table,
        (fresh_metal_ejecta + old_metal_ejecta)[:, 0],
        sn_reduction,
        masses,
    )
    ejected_dust = (fresh_dust_ejecta * sfr_vals *
                    imf_vals * d_masses).sum(axis=0)

    return ejected_gas, ejected_metal, ejected_dust


def Gauss_Kronrod_ejecta(
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
    calculate the gas, metals, and dust emmitted from dying stars, given
    a function for mass of stellar remnants as well as output tables for
    metal and dust yields. In essence, convolves the past SFR with the IMF
    and a yield function to find how much gas/metal/dust is beind shot out now
    requires:
        - \"dust_yields\": table where each row gives a mass in Msol, and the
                           dust *created*, not recycled, when such a star dies
        - \"metal_yields\": table where each row gives a mass in Msol, followed
                            by several entries giving the metals created when
                            such a star dies, ordered the same as init_metals,
                            repeated for each metallicity level
        - \"yield_table_z_cutoffs\": list giving the cutoff for each
                                     metallicity level in the metal_yields table
        - \"sn_dust_reduction\": factor which divides dust created in supernovae
        - \"stellar_lifetimes\": table where each row gives, in order
                                 the mass of a star in Msol, the lifetime
                                 of such a star in Gyrs in a low metallicity
                                 (Z < 0.008) environment, and the lifetime in
                                 a high metallicity (Z >= 0.008) environment
    Uses Gauss-Kronrod integration to (ideally) lower the workload and also
    give an error estimate. If the error is too high, subdivides interval.
    """

    # store all model params for easier passing to subroutines
    dust_yield_table = model_params["dust_yields"]
    metal_yield_table = model_params["metal_yields"]
    metallicity_cutoffs = model_params["yield_table_z_cutoffs"]
    sn_reduction = model_params["sn_dust_reduction"]
    stellar_lifetimes = model_params["stellar_lifetimes"]

    # grab everything precomputable, and precompute it if not
    # masses we sample, imf at each mass, gas ejecta at each mass,
    # as well as weights
    try:
        masses = cache["ejecta_masses"]
        ejecta = cache["ejecta_vals"]
        imf_vals = cache["imf_values"]
        gauss_weights = cache["gauss_weights"]
        kronrod_weights = cache["kronrod_weights"]

    except KeyError:

        cache["ejecta_masses"] = ((sample_points_pre + 1) / 2) * (119.2) + 0.8
        cache["ejecta_subdivisions"] = 1
        masses = cache["ejecta_masses"]
        remnants = remnant_mass(masses)
        cache["ejecta_vals"] = masses - remnants
        cache["imf_values"] = imf(masses)
        cache["gauss_weights"] = gauss_weights_pre * 59.6
        cache["kronrod_weights"] = kronrod_weights_pre * 59.6

        ejecta = cache["ejecta_vals"]
        imf_vals = cache["imf_values"]
        gauss_weights = cache["gauss_weights"]
        kronrod_weights = cache["kronrod_weights"]

    # determine if high or low metallicity lifetimes are to be used
    if (mmetal[0] / mgas[0]) <= 0.008:
        metallicity = "low"

    else:
        metallicity = "high"

    while True:

        lifetimes = life_from_mass_vec(masses, stellar_lifetimes, metallicity)

        imf_vals = where(t > lifetimes, imf_vals, 0)

        # create arrays for historical metallicity and sfr
        z_at_birth = metal_hist(t - lifetimes) / gas_hist(t - lifetimes)
        sfr_vals = sfr_hist(t - lifetimes)

        # calculate all our ejecta
        ejected_gas_k = (ejecta * sfr_vals * imf_vals *
                         kronrod_weights).sum(axis=0)
        ejected_gas_g = (ejecta * sfr_vals * imf_vals *
                         gauss_weights).sum(axis=0)

        fresh_metal_ejecta = fresh_metals(
            metal_yield_table, metallicity_cutoffs, masses, mmetal / mgas[0]
        )
        old_metal_ejecta = ejecta[:, None] * z_at_birth
        ejected_metal_k = (
            (fresh_metal_ejecta + old_metal_ejecta)
            * sfr_vals[:, None]
            * imf_vals[:, None]
            * kronrod_weights[:, None]
        ).sum(axis=0)
        ejected_metal_g = (
            (fresh_metal_ejecta + old_metal_ejecta)
            * sfr_vals[:, None]
            * imf_vals[:, None]
            * gauss_weights[:, None]
        ).sum(axis=0)

        fresh_dust_ejecta = fresh_dust(
            dust_yield_table,
            (fresh_metal_ejecta + old_metal_ejecta)[:, 0],
            sn_reduction,
            masses,
        )
        ejected_dust_k = (
            fresh_dust_ejecta * sfr_vals * imf_vals * kronrod_weights
        ).sum(axis=0)
        ejected_dust_g = (fresh_dust_ejecta * sfr_vals * imf_vals * gauss_weights).sum(
            axis=0
        )

        err_gas = abs(ejected_gas_k - ejected_gas_g)
        err_metal = abs(ejected_metal_k - ejected_metal_g)
        err_dust = abs(ejected_dust_k - ejected_dust_g)

        err = hstack((err_gas, err_metal, err_dust)) / (
            1
            + 1e-3
            * maximum(
                hstack((ejected_gas_k, ejected_metal_k, ejected_dust_k)),
                hstack((ejected_gas_g, ejected_metal_g, ejected_dust_g)),
            )
        )

        err = sqrt((err**2).mean())

        if err <= 1 or err != err:
            try:
                cache["integral_accuracy"][t] = cache["ejecta_subdivisions"]
            except KeyError:
                cache["integral_accuracy"] = {}
                cache["integral_accuracy"][t] = cache["ejecta_subdivisions"]

            return ejected_gas_k, ejected_metal_k, ejected_dust_k
        else:

            cache["ejecta_subdivisions"] *= 2
            ints = cache["ejecta_subdivisions"]

            mesh = array([0.8 + (i * 119.2 / ints) for i in range(ints + 1)])

            cache["ejecta_masses"] = []
            cache["gauss_weights"] = []
            cache["kronrod_weights"] = []

            for i in range(0, ints):
                cache["ejecta_masses"].extend(
                    ((sample_points_pre + 1) / 2) *
                    (mesh[i + 1] - mesh[i]) + mesh[i]
                )
                cache["gauss_weights"].extend(
                    gauss_weights_pre * (mesh[i + 1] - mesh[i]) / 2
                )
                cache["kronrod_weights"].extend(
                    kronrod_weights_pre * (mesh[i + 1] - mesh[i]) / 2
                )

            cache["ejecta_masses"] = array(cache["ejecta_masses"])
            cache["gauss_weights"] = array(cache["gauss_weights"])
            cache["kronrod_weights"] = array(cache["kronrod_weights"])

            masses = cache["ejecta_masses"]
            remnants = remnant_mass(masses)
            cache["ejecta_vals"] = masses - remnants
            cache["imf_values"] = imf(masses)

            ejecta = cache["ejecta_vals"]
            imf_vals = cache["imf_values"]
            gauss_weights = cache["gauss_weights"]
            kronrod_weights = cache["kronrod_weights"]


# Sample points and weights for 15-point Gauss-Kronrod integration.
# Based on integrating on [-1,1], needs to be rescaled to new bounds
"""
sample_points_pre = array(
    [
        0.991455371120813,
        0.949107912342759,
        0.864864423359769,
        0.741531185599394,
        0.586087235467691,
        0.405845151377397,
        0.207784955007898,
        0.000000000000000,
        -0.207784955007898,
        -0.405845151377397,
        -0.586087235467691,
        -0.741531185599394,
        -0.864864423359769,
        -0.949107912342759,
        -0.991455371120813,
    ]
)

gauss_weights_pre = array(
    [
        0,
        0.129484966168870,
        0,
        0.279705391489277,
        0,
        0.381830050505119,
        0,
        0.417959183673469,
        0,
        0.381830050505119,
        0,
        0.279705391489277,
        0,
        0.129484966168870,
        0,
    ]
)

kronrod_weights_pre = array(
    [
        0.022935322010529,
        0.063092092629979,
        0.104790010322250,
        0.129484966168870,
        0.169004726639267,
        0.190350578064785,
        0.204432940075298,
        0.209482141084728,
        0.204432940075298,
        0.190350578064785,
        0.169004726639267,
        0.129484966168870,
        0.104790010322250,
        0.063092092629979,
        0.022935322010529,
    ]
)
"""

# Sample points and weights for 61-point Gauss-Kronrod integration.
# Based on integrating on [-1,1], needs to be rescaled to new bounds
sample_points_pre = array(
    [
        -9.994844100504906375713258957058108e-01,
        -9.968934840746495402716300509186953e-01,
        -9.916309968704045948586283661094857e-01,
        -9.836681232797472099700325816056628e-01,
        -9.731163225011262683746938684237069e-01,
        -9.600218649683075122168710255817977e-01,
        -9.443744447485599794158313240374391e-01,
        -9.262000474292743258793242770804740e-01,
        -9.055733076999077985465225589259583e-01,
        -8.825605357920526815431164625302256e-01,
        -8.572052335460610989586585106589439e-01,
        -8.295657623827683974428981197325019e-01,
        -7.997278358218390830136689423226832e-01,
        -7.677774321048261949179773409745031e-01,
        -7.337900624532268047261711313695276e-01,
        -6.978504947933157969322923880266401e-01,
        -6.600610641266269613700536681492708e-01,
        -6.205261829892428611404775564311893e-01,
        -5.793452358263616917560249321725405e-01,
        -5.366241481420198992641697933110728e-01,
        -4.924804678617785749936930612077088e-01,
        -4.470337695380891767806099003228540e-01,
        -4.004012548303943925354762115426606e-01,
        -3.527047255308781134710372070893739e-01,
        -3.040732022736250773726771071992566e-01,
        -2.546369261678898464398051298178051e-01,
        -2.045251166823098914389576710020247e-01,
        -1.538699136085835469637946727432559e-01,
        -1.028069379667370301470967513180006e-01,
        -5.147184255531769583302521316672257e-02,
        0.000000000000000000000000000000000e00,
        5.147184255531769583302521316672257e-02,
        1.028069379667370301470967513180006e-01,
        1.538699136085835469637946727432559e-01,
        2.045251166823098914389576710020247e-01,
        2.546369261678898464398051298178051e-01,
        3.040732022736250773726771071992566e-01,
        3.527047255308781134710372070893739e-01,
        4.004012548303943925354762115426606e-01,
        4.470337695380891767806099003228540e-01,
        4.924804678617785749936930612077088e-01,
        5.366241481420198992641697933110728e-01,
        5.793452358263616917560249321725405e-01,
        6.205261829892428611404775564311893e-01,
        6.600610641266269613700536681492708e-01,
        6.978504947933157969322923880266401e-01,
        7.337900624532268047261711313695276e-01,
        7.677774321048261949179773409745031e-01,
        7.997278358218390830136689423226832e-01,
        8.295657623827683974428981197325019e-01,
        8.572052335460610989586585106589439e-01,
        8.825605357920526815431164625302256e-01,
        9.055733076999077985465225589259583e-01,
        9.262000474292743258793242770804740e-01,
        9.443744447485599794158313240374391e-01,
        9.600218649683075122168710255817977e-01,
        9.731163225011262683746938684237069e-01,
        9.836681232797472099700325816056628e-01,
        9.916309968704045948586283661094857e-01,
        9.968934840746495402716300509186953e-01,
        9.994844100504906375713258957058108e-01,
    ]
)


kronrod_weights_pre = array(
    [
        1.389013698677007624551591226759700e-03,
        3.890461127099884051267201844515503e-03,
        6.630703915931292173319826369750168e-03,
        9.273279659517763428441146892024360e-03,
        1.182301525349634174223289885325059e-02,
        1.436972950704580481245143244358001e-02,
        1.692088918905327262757228942032209e-02,
        1.941414119394238117340895105012846e-02,
        2.182803582160919229716748573833899e-02,
        2.419116207808060136568637072523203e-02,
        2.650995488233310161060170933507541e-02,
        2.875404876504129284397878535433421e-02,
        3.090725756238776247288425294309227e-02,
        3.298144705748372603181419101685393e-02,
        3.497933802806002413749967073146788e-02,
        3.688236465182122922391106561713597e-02,
        3.867894562472759295034865153228105e-02,
        4.037453895153595911199527975246811e-02,
        4.196981021516424614714754128596976e-02,
        4.345253970135606931683172811707326e-02,
        4.481480013316266319235555161672324e-02,
        4.605923827100698811627173555937358e-02,
        4.718554656929915394526147818109949e-02,
        4.818586175708712914077949229830459e-02,
        4.905543455502977888752816536723817e-02,
        4.979568342707420635781156937994233e-02,
        5.040592140278234684089308565358503e-02,
        5.088179589874960649229747304980469e-02,
        5.122154784925877217065628260494421e-02,
        5.142612853745902593386287921578126e-02,
        5.149472942945156755834043364709931e-02,
        5.142612853745902593386287921578126e-02,
        5.122154784925877217065628260494421e-02,
        5.088179589874960649229747304980469e-02,
        5.040592140278234684089308565358503e-02,
        4.979568342707420635781156937994233e-02,
        4.905543455502977888752816536723817e-02,
        4.818586175708712914077949229830459e-02,
        4.718554656929915394526147818109949e-02,
        4.605923827100698811627173555937358e-02,
        4.481480013316266319235555161672324e-02,
        4.345253970135606931683172811707326e-02,
        4.196981021516424614714754128596976e-02,
        4.037453895153595911199527975246811e-02,
        3.867894562472759295034865153228105e-02,
        3.688236465182122922391106561713597e-02,
        3.497933802806002413749967073146788e-02,
        3.298144705748372603181419101685393e-02,
        3.090725756238776247288425294309227e-02,
        2.875404876504129284397878535433421e-02,
        2.650995488233310161060170933507541e-02,
        2.419116207808060136568637072523203e-02,
        2.182803582160919229716748573833899e-02,
        1.941414119394238117340895105012846e-02,
        1.692088918905327262757228942032209e-02,
        1.436972950704580481245143244358001e-02,
        1.182301525349634174223289885325059e-02,
        9.273279659517763428441146892024360e-03,
        6.630703915931292173319826369750168e-03,
        3.890461127099884051267201844515503e-03,
        1.389013698677007624551591226759700e-03,
    ]
)

gauss_weights_pre = array(
    [
        0,
        7.968192496166605615465883474673622e-03,
        0,
        1.846646831109095914230213191204727e-02,
        0,
        2.878470788332336934971917961129204e-02,
        0,
        3.879919256962704959680193644634769e-02,
        0,
        4.840267283059405290293814042280752e-02,
        0,
        5.749315621761906648172168940205613e-02,
        0,
        6.597422988218049512812851511596236e-02,
        0,
        7.375597473770520626824385002219073e-02,
        0,
        8.075589522942021535469493846052973e-02,
        0,
        8.689978720108297980238753071512570e-02,
        0,
        9.212252223778612871763270708761877e-02,
        0,
        9.636873717464425963946862635180987e-02,
        0,
        9.959342058679526706278028210356948e-02,
        0,
        1.017623897484055045964289521685540e-01,
        0,
        1.028526528935588403412856367054150e-01,
        0,
        1.028526528935588403412856367054150e-01,
        0,
        1.017623897484055045964289521685540e-01,
        0,
        9.959342058679526706278028210356948e-02,
        0,
        9.636873717464425963946862635180987e-02,
        0,
        9.212252223778612871763270708761877e-02,
        0,
        8.689978720108297980238753071512570e-02,
        0,
        8.075589522942021535469493846052973e-02,
        0,
        7.375597473770520626824385002219073e-02,
        0,
        6.597422988218049512812851511596236e-02,
        0,
        5.749315621761906648172168940205613e-02,
        0,
        4.840267283059405290293814042280752e-02,
        0,
        3.879919256962704959680193644634769e-02,
        0,
        2.878470788332336934971917961129204e-02,
        0,
        1.846646831109095914230213191204727e-02,
        0,
        7.968192496166605615465883474673622e-03,
        0,
    ]
)
