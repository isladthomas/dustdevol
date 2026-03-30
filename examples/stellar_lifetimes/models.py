from dustdevol.BEDE2 import (
    sfr_from_efficiency,
    bursty_sfr_from_efficiency,
    BEDE_inflow,
    Nelson_outflow,
    THEMIS_grain_growth,
    dust_destruction,
    stellar_ejecta,
    BEDE_recycling,
)
from dustdevol.DeVis2017 import stellar_ejecta as old_stellar_ejecta
from dustdevol.DeVis2017 import dust_destruction as old_dust_destruction
from dustdevol.generic import (
    fp,
    fp_array,
    fp_zeros,
    stellar_lifetimes,
    stellar_lifetimes_pchip,
    stellar_lifetimes_lin,
    stellar_lifetimes_nn,
    h_stellar_lifetimes,
    h_stellar_lifetimes_pchip,
    h_stellar_lifetimes_lin,
    h_stellar_lifetimes_nn,
    S92,
    TF01,
    KA18_high_nn,
    KA18_high_lin,
    LC18_R150_nn,
    LC18_R150_lin,
    LN2018_nn,
    LN2018_lin,
    AGB_SN_yields,
    S2020,
)
from dustdevol.imf import salp
from copy import deepcopy
from numpy import array, full, dtype

yields_nn = AGB_SN_yields(KA18_high_nn, LC18_R150_nn, 3)

yields_lin = AGB_SN_yields(KA18_high_lin, LC18_R150_lin, 3)

reference = {
    "time_start": fp(0),
    "time_end": fp(13.8),
    "sfr_model": sfr_from_efficiency,
    "imf": salp,
    "inflow_model": BEDE_inflow,
    "outflow_model": Nelson_outflow,
    "recycling_model": BEDE_recycling,
    "grain_growth_model": THEMIS_grain_growth,
    "destruction_model": dust_destruction,
    "ejecta_model": stellar_ejecta,
    "init_gas": fp_array([0.5e10]),
    "init_star": fp_array([0]),
    "init_metal": fp_array([0, 0, 0]),
    "init_dust": fp_array([0]),
    "model_params": {
        "star_formation_efficiency": fp(1),
        "total_inflow": fp(0.5e10),
        "infall_time": fp(2),
        "tot_infall_time": fp(13.79),
        "inflow_metal": fp(0),
        "inflow_dust": fp(0),
        "outflow_metal": fp(1),
        "outflow_dust": fp(1),
        "recycling_scaling": fp(1),
        "IGM_loss": fp(0.2),
        "grain_growth_epsilon_diffuse": fp(5),
        "grain_growth_epsilon_cloud": fp(4000),
        "available_metals": fp(0.2),
        "cold_fraction": fp(0.5),
        "sn_dust_reduction": fp(5),
        "sn_destruction": fp(15),
        "type_Ia_ratio": fp(0),
        "photofrag_efficiency": fp(0.05),
        "silicate_fraction": fp(0.1),
        "stellar_lifetimes": stellar_lifetimes,
        "dust_yields": TF01,
        "metal_yields": yields_nn,
        "type_Ia_dust_yields": fp_zeros((2, 2)),
        "type_Ia_metal_yields": LN2018_nn,
        "type_Ia_delays": S2020,
        "type_Ia_probability": 0,
    },
    "absolute_tolerance": 1,
    "relative_tolerance": 1e-3,
}

model_grid = array(
    [
        [
            [
                [
                    [
                        [deepcopy(reference) for birth_death in range(2)]
                        for type_Ia in range(2)
                    ]
                    for interpolation in range(3)
                ]
                for h_or_c in range(2)
            ]
            for yield_interpolation in range(2)
        ]
        for Ia_interpolation in range(2)
    ]
)
name_dict = {
    "Ia_interpolation": "nn",
    "yield_interpolation": "nn",
    "death_point": "carbon",
    "lifetime_interpolation": "cube",
    "type_Ia": "off",
    "birth_death_calc": "birth",
}

model_names = array(
    [
        [
            [
                [
                    [
                        [deepcopy(name_dict) for birth_death in range(2)]
                        for type_Ia in range(2)
                    ]
                    for interpolation in range(3)
                ]
                for h_or_c in range(2)
            ]
            for yield_interpolation in range(2)
        ]
        for Ia_interpolation in range(2)
    ]
)

for model, name in zip(model_grid[1].flatten(), model_names[1].flatten()):
    model["model_params"]["type_Ia_metal_yields"] = LN2018_lin
    name["Ia_interpolation"] = "lin"

for model, name in zip(model_grid[:, 1].flatten(), model_names[:, 1].flatten()):
    model["model_params"]["metal_yields"] = yields_lin
    name["yield_interpolation"] = "lin"

for model, name in zip(
    model_grid[:, :, 0, 1].flatten(), model_names[:, :, 0, 1].flatten()
):
    model["model_params"]["stellar_lifetimes"] = stellar_lifetimes_lin
    name["lifetime_interpolation"] = "lin"
for model, name in zip(
    model_grid[:, :, 0, 2].flatten(), model_names[:, :, 0, 2].flatten()
):
    model["model_params"]["stellar_lifetimes"] = stellar_lifetimes_nn
    name["lifetime_interpolation"] = "nn"

for model, name in zip(
    model_grid[:, :, 1, 0].flatten(), model_names[:, :, 1, 0].flatten()
):
    model["model_params"]["stellar_lifetimes"] = h_stellar_lifetimes
    name["death_point"] = "hydrogen"
for model, name in zip(
    model_grid[:, :, 1, 1].flatten(), model_names[:, :, 1, 1].flatten()
):
    model["model_params"]["stellar_lifetimes"] = h_stellar_lifetimes_lin
    name["lifetime_interpolation"] = "lin"
    name["death_point"] = "hydrogen"
for model, name in zip(
    model_grid[:, :, 1, 2].flatten(), model_names[:, :, 1, 2].flatten()
):
    model["model_params"]["stellar_lifetimes"] = h_stellar_lifetimes_nn
    name["lifetime_interpolation"] = "nn"
    name["death_point"] = "hydrogen"

for model, name in zip(
    model_grid[:, :, :, :, 1].flatten(), model_names[:, :, :, :, 1].flatten()
):
    model["model_params"]["type_Ia_probability"] = 0.001 / 2.828956424049581
    name["type_Ia"] = "on"

for model, name in zip(
    model_grid[:, :, :, :, :, 1].flatten(), model_names[:, :,
                                                        :, :, :, 1].flatten()
):
    model["ejecta_model"] = old_stellar_ejecta
    model["destruction_model"] = old_dust_destruction
    name["birth_death_calc"] = "death"
