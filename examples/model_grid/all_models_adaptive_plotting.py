from numpy import load, maximum, exp, minimum
import matplotlib.pyplot as plt
from dustdevol.generic import z_at_t
from dustdevol.BEDE2.gas import mass_loading

with load("outputs/reference_corrected_recycling.npz") as d1, load(
    "outputs/reference.npz"
) as d2, load(
    "outputs/reference_original_model.npz"
) as d3:
    plt.figure(1)
    plt.plot(
        d1["times"],
        d1["gas_masses"],
        color="xkcd:red",
        linestyle="solid",
    )
    plt.plot(
        d1["times"],
        d1["star_masses"],
        color="xkcd:blue",
        linestyle="solid",
    )
    plt.plot(
        d1["times"],
        d1["sfr"],
        color="xkcd:purple",
        linestyle="solid",
    )

    plt.plot(
        d3["time"],
        d3["mgas"],
        color="xkcd:light red",
        linestyle="dashed",
    )
    plt.plot(
        d3["time"],
        d3["mstars"],
        color="xkcd:light blue",
        linestyle="dashed",
    )
    plt.plot(
        d3["time"],
        d3["sfr"] * 1e9,
        color="xkcd:light purple",
        linestyle="dashed",
    )

    plt.figure(2)
    plt.plot(
        d1["times"],
        d1["recycling"],
        color="xkcd:green",
        linestyle="solid",
    )
    plt.plot(
        d1["times"],
        d1["outflow"],
        color="xkcd:red",
        linestyle="solid",
    )
    plt.plot(
        d1["times"],
        d1["inflow"],
        color="xkcd:blue",
        linestyle="solid",
    )
    plt.plot(
        d1["times"],
        d1["sfr"],
        color="xkcd:purple",
        linestyle="solid",
    )
    plt.plot(
        d1["times"],
        d1["gas_masses"] * 0.5 / 0.03,
        color="xkcd:dark grey",
        linestyle="solid",
    )
    plt.plot(
        d3["time"][:-1],
        d3["mg_recycled"],
        color="xkcd:light green",
        linestyle="dashed",
    )
    plt.plot(
        d3["time"][:-1],
        d3["mg_outflow"],
        color="xkcd:light red",
        linestyle="dashed",
    )
    plt.plot(
        d3["time"][:-1],
        d3["mg_inflow"],
        color="xkcd:light blue",
        linestyle="dashed",
    )
    plt.plot(
        d3["time"],
        d3["sfr"] * 1e9,
        color="xkcd:light purple",
        linestyle="dashed",
    )
    plt.plot(
        d3["time"],
        d3["mgas"] * 0.5 / 0.03,
        color="xkcd:grey",
        linestyle="dashed",
    )

plt.figure(1)
plt.suptitle("Reference Model")
plt.title("Component Mass Overview")
plt.ylabel("Component Mass (Msol)")
plt.xlabel("Time (Gyr)")
plt.yscale("log")
plt.legend(["gas", "stars", "sfr"])
plt.savefig("plots/reference_corrected_outflows.eps")
plt.clf()

plt.figure(2)
plt.suptitle("Reference Model")
plt.title("Component Mass Overview")
plt.ylabel("Component Derivative (Msol/Gyr)")
plt.xlabel("Time (Gyr)")
plt.yscale("log")
plt.legend(["recycling", "outflow", "inflow", "sfr", "maximum outflow"])
plt.savefig("plots/reference_corrected_outflows_diffs.eps")
plt.clf()
