from dustdevol.evolve_FCRK import evolve_2o_FC
from models import reference
from numpy import savez_compressed, diff, array
import logging
from timeit import default_timer as timer
from chemevol import ChemModel

logging.captureWarnings(True)
logging.basicConfig(filename="Warnings.log", level=logging.WARNING)

start = timer()
results = evolve_2o_FC(**reference)
end = timer()

time = end - start

print("Finished in {} seconds".format(time))
print("Only took {} steps!".format(len(results["times"])))
print("(out of {} attempted steps...)".format(
    results["cache"]["attempted_steps"]))
print("Smallest step was {} Gyr".format(diff(results["times"]).min()))
a = array(list(results["cache"]["recycling"].items()))
a = a[a[:, 0].argsort()]
b = array(list(results["cache"]["outflow"].items()))
b = b[b[:, 0].argsort()]
c = array(list(results["cache"]["inflow"].items()))
c = c[c[:, 0].argsort()]

output = {
    "times": results["times"],
    "gas_masses": results["gas_masses"],
    "star_masses": results["star_masses"],
    "metal_masses": results["metal_masses"],
    "dust_masses": results["dust_masses"],
    "dgas_masses": results["dgas_masses"],
    "dstar_masses": results["dstar_masses"],
    "dmetal_masses": results["dmetal_masses"],
    "ddust_masses": results["ddust_masses"],
    "sfr": results["sfr"],
    "recycling": a[:, 1],
    "outflow": b[:, 1],
    "inflow": c[:, 1]
}
savez_compressed("outputs/reference_corrected_recycling", **output)

item = {
    "name": "Interpolate",
    "gasmass_init": 1e10,
    "starmass_init": 0,
    "dustmass_init": 0,
    "Z_init": 0,
    "SFH": "average.sfe",
    "add_bursts": False,
    "t_end": 13.8,
    "t_start": 0.001,
    "gamma": 0,
    "IMF_fn": "Salp",
    "dust_source": "ALL",
    "cold_gas_fraction": 0.5,
    "use_THEMIS": True,
    "delta_lims_fresh": 0.15,
    "reduce_sn_dust": {"on": True, "factor": 5},
    "destroy": {"on": True, "mass": 15},
    "fragmentgrains": {"on": True, "tau": 0.05},
    "effective_snrate_factor": 0.36,
    "graingrowth": 4000,
    "graingrowth2": 5,
    "inflows": {
        "on": True,
        "mass": 0,
        "metals": 0,
        "isxSFR": False,
        "xSFR": 1,
        "dust": 0,
    },
    "outflows": {
        "on": True,
        "metals": True,
        "dust": True,
        "reduce": 1,
    },
    "recycle": {"on": True, "esc_prob_perGyr": 0.2, "reaccr_time_factor": 1.0},
    "available_metal_fraction": 0.2,
    "SNyield": "LC18_R150",
    "AGByield": "KA18_high",
    "totyields": True,
    "isotopes": ["Z", "O", "N"],
    "Pristine_isotope_fractions": [1.0, 0.435, 0.055],
    "redshift": "interpolate",
}

start = timer()
ch = ChemModel(**item)

print("Now evaluating " + item["name"])

snrate = ch.supernova_rate()
all_results = ch.gas_metal_dust_mass(snrate)
end = timer()
print(end - start)

params = {
    "time": all_results[:, 0],
    "z": all_results[:, 1],
    "mgas": all_results[:, 2],
    "mstars": all_results[:, 3],
    "metals": all_results[:, 24],
    "oxygen": all_results[:, 25],
    "nitrogen": all_results[:, 26],
    "mdust": all_results[:, 5],
    "sfr": all_results[:, 7],
    "mg_recycled": all_results[:-1, 18] / diff(all_results[:, 0]),
    "mg_outflow": all_results[:-1, 17] / diff(all_results[:, 0]),
    "mg_inflow": all_results[:-1, 19] / diff(all_results[:, 0]),
}

savez_compressed("outputs/reference_original_model", **params)
