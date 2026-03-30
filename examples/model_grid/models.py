from itertools import product
from dustdevol.BEDE2 import (
    sfr_from_efficiency,
    bursty_sfr_from_efficiency,
    BEDE_inflow,
    Nelson_outflow,
    THEMIS_grain_growth,
    THEMIS_dust_destruction,
    stellar_ejecta,
    BEDE_recycling,
)
from dustdevol.generic import (
    fp,
    fp_array,
    fp_zeros,
    stellar_lifetimes_lin,
    TF01,
    KA18_high_lin,
    LC18_R150_lin,
    AGB_SN_yields,
    LN2018_lin,
    S2020,
)
from dustdevol.imf import salp
from dustdevol.dust_models import (
    Asano_gg,
    BEDE_gg,
    DeVis_gg,
    Asano_dd,
    DeVis_dd,
    Priestley_dd,
)
from copy import deepcopy

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
    "destruction_model": THEMIS_dust_destruction,
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
        "photofrag_efficiency": fp(0.05),
        "silicate_fraction": fp(0.1),
        "stellar_lifetimes": stellar_lifetimes_lin,
        "dust_yields": TF01,
        "metal_yields": yields_lin,
        "type_Ia_dust_yields": fp_zeros((2, 2)),
        "type_Ia_metal_yields": LN2018_lin,
        "type_Ia_delays": S2020,
        "type_Ia_probability": 0.001 / 2.828956424049581,
    },
    "absolute_tolerance": 1,
    "relative_tolerance": 1e-3,
}


# Source - https://stackoverflow.com/a/65392983
# loop over dictionary parameters python
# Posted by ssp, modified by community. See post 'Timeline' for change history
# Retrieved 2026-03-14, License - CC BY-SA 4.0
def dict_configs(d):
    for vcomb in product(*d.values()):
        yield dict(zip(d.keys(), vcomb))


model_grid = {
    "grain_growth_model": [Asano_gg, BEDE_gg],
    "destruction_model": [Asano_dd, DeVis_dd, Priestley_dd],
}
model_parameters = {
    "sn_dust_reduction": [1, 5, 20, 80],
    "sn_destruction": [0, 2000, 4000],
    "photofrag_efficiency": [0.0, 0.05, 0.5, 1, 5],
    "grain_growth_epsilon_cloud": [1000, 2000, 4000, 8000, 16000],
    "grain_growth_epsilon_diffuse": [0, 5, 10],
    "available_metals": [0.2, 0.3, 0.4],
}
