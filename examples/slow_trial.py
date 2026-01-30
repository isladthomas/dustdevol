from chemevol import ChemModel
from dustdevol.evolve import evolve_sfr
from dustdevol.imf import chab
import dustdevol.generic as g
from dustdevol.DeVis2017 import (
    xSFR_inflow,
    xSFR_outflow,
    grain_growth,
    dust_destruction,
    stellar_ejecta,
    fast_ejecta_lin,
    fast_ejecta_log,
)
from numpy import arange, zeros

ch = ChemModel(
    **{
        "name": "Model_I_test",
        "gasmass_init": 4e10,
        "SFH": "Milkyway_2017.sfh",
        "t_end": 20.0,
        "gamma": 0,
        "IMF_fn": "Chab",
        "dust_source": "SN+LIMS",
        "reduce_sn_dust": False,
        "destroy": False,
        "inflows": {"metals": 0.0, "xSFR": 0, "dust": 0},
        "outflows": {"metals": False, "xSFR": 0, "dust": False},
        "cold_gas_fraction": 0.5,
        "epsilon_grain": 0,
        "destruct": 0,
    },
)

results_slow = evolve_sfr(
    g.fp(0),
    g.fp(20),
    ch.sfh[:,0][ch.sfh[:,0] < g.fp(2)],
    g.sfr_from_file,
    chab,
    xSFR_inflow,
    xSFR_outflow,
    g.off,
    grain_growth,
    dust_destruction,
    stellar_ejecta,
    [4e10],
    [0],
    [0, 0],
    [0],
    {
        "sfr_file": "Milkyway_2017.sfh",
        "sn_dust_reduction": 1,
        "sn_destruction": 0,
        "inflow_xSFR": 0,
        "outflow_xSFR": 0,
        "cold_fraction": 0.5,
        "grain_growth_epsilon": 0,
        "stellar_lifetimes": g.S92,
        "dust_yields": g.TF01,
        "metal_yields": g.vdHG97_M92_yields,
        "yield_table_z_cutoffs": g.vdHG97_M92_cutoffs,
        "sfr_interp_order": 0,
    },
)
