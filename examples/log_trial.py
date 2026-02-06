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
from timeit import default_timer as timer

start = timer()
results_log = evolve_sfr(
    g.fp(0),
    g.fp(20),
    arange(0.05, 20, 0.05, dtype=g.fp),
    g.sfr_from_file,
    chab,
    xSFR_inflow,
    xSFR_outflow,
    g.off,
    grain_growth,
    dust_destruction,
    fast_ejecta_log,
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
    },
)
end = timer()

print(end - start)
