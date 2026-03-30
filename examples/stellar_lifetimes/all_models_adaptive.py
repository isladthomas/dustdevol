from dustdevol.evolve_FCRK import evolve_2o_FC
from models import model_grid, model_names
from numpy import savez_compressed, diff, array
import logging
from timeit import default_timer as timer
from chemevol import ChemModel

logging.captureWarnings(True)
logging.basicConfig(filename="Warnings.log", level=logging.WARNING)

for model, name in zip(model_grid.flatten(), model_names.flatten()):
    print("Now Running Model " + "_".join(name.values()))
    start = timer()
    results = evolve_2o_FC(**model)
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
        "gas_func": results["gas_func"],
        "star_func": results["star_func"],
        "metal_func": results["metal_func"],
        "dust_func": results["dust_func"],
        "sfr": results["sfr"],
    }
    savez_compressed("outputs/" + "_".join(name.values()), **output)
