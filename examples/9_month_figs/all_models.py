from dustdevol.evolve_FCRK import evolve_2o_FC
from models import reference, model_grid, dict_configs
from numpy import savez_compressed, diff, array
import logging
from timeit import default_timer as timer
from chemevol import ChemModel
from copy import deepcopy

logging.captureWarnings(True)
logging.basicConfig(filename="Warnings.log", level=logging.WARNING)

running = deepcopy(reference)

start = timer()
results = evolve_2o_FC(**reference)
end = timer()

time = end - start

print("Finished in {} seconds".format(time))
print("Only took {} steps!".format(len(results["times"])))
print(
    "(out of {} attempted steps...)".format(
        results["cache"]["attempted_steps"])
)
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

savez_compressed(
    "outputs/ref_lifetime_updated.npz",
    **output
)

running["model_params"].update({"type_Ia_probability": 0})

start = timer()
results = evolve_2o_FC(**running)
end = timer()

time = end - start

print("Finished in {} seconds".format(time))
print("Only took {} steps!".format(len(results["times"])))
print(
    "(out of {} attempted steps...)".format(
        results["cache"]["attempted_steps"])
)
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

savez_compressed(
    "outputs/Ia_off_lifetime_updated.npz",
    **output
)


for model in dict_configs(model_grid):

    running.update(model)

    start = timer()
    results = evolve_2o_FC(**running)
    end = timer()

    time = end - start

    print("Finished in {} seconds".format(time))
    print("Only took {} steps!".format(len(results["times"])))
    print(
        "(out of {} attempted steps...)".format(
            results["cache"]["attempted_steps"])
    )
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
        "gg_efficiency": array(
            [results["cache"]["gg_efficiency"][t]
                for t in results["times"]]
        )[:, None],
        "gg_diffuse_timescale": array(
            [results["cache"]["gg_diffuse_timescale"][t]
                for t in results["times"]]
        )[:, None],
        "gg_cloud_timescale": array(
            [results["cache"]["gg_cloud_timescale"][t]
                for t in results["times"]]
        )[:, None],
        "dd_efficiency": array(
            [results["cache"]["dd_efficiency"][t]
                for t in results["times"]]
        )[:, None],
        "dd_frag_timescale": array(
            [results["cache"]["dd_frag_timescale"][t]
                for t in results["times"]]
        )[:, None],
        "dd_des_timescale": array(
            [results["cache"]["dd_des_timescale"][t]
                for t in results["times"]]
        )[:, None],
    }
    savez_compressed(
        "outputs/"
        + "_".join([x.__name__ for x in model.values()])
        + "_"
        + "_lifetime_updated.npz",
        **output
    )
