from dustdevol.evolve_FCRK import evolve_2o_FC
from models import reference, reference_Ia
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
print("(out of {} attempted steps...)".format(results["cache"]["attempted_steps"]))
print("Smallest step was {} Gyr".format(diff(results["times"]).min()))

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
}
savez_compressed("outputs/reference", **output)

start = timer()
results = evolve_2o_FC(**reference_Ia)
end = timer()

time = end - start

print("Finished in {} seconds".format(time))
print("Only took {} steps!".format(len(results["times"])))
print("(out of {} attempted steps...)".format(results["cache"]["attempted_steps"]))
print("Smallest step was {} Gyr".format(diff(results["times"]).min()))

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
}
savez_compressed("outputs/reference_Ia", **output)
