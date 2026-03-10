from numpy import load, maximum, exp, minimum
import matplotlib.pyplot as plt
from dustdevol.generic import z_at_t
from dustdevol.BEDE2.gas import mass_loading

with load("outputs/reference.npz") as d1, load("outputs/reference_Ia.npz") as d2:
    plt.plot(
        d1["times"],
        d1["gas_masses"],
        color="xkcd:green",
        linestyle="solid",
    )
    plt.plot(
        d1["times"],
        d1["star_masses"],
        color="xkcd:purple",
        linestyle="solid",
    )
    plt.plot(
        d1["times"],
        d1["metal_masses"][:, 0],
        color="xkcd:blue",
        linestyle="solid",
    )
    plt.plot(
        d1["times"],
        d1["metal_masses"][:, 1],
        color="xkcd:red",
        linestyle="solid",
    )
    plt.plot(
        d1["times"],
        d1["dust_masses"],
        color="xkcd:shit brown",
        linestyle="solid",
    )

    plt.plot(
        d2["times"],
        d2["gas_masses"],
        color="xkcd:light green",
        linestyle="dashed",
    )
    plt.plot(
        d2["times"],
        d2["star_masses"],
        color="xkcd:light purple",
        linestyle="dashed",
    )
    plt.plot(
        d2["times"],
        d2["metal_masses"][:, 0],
        color="xkcd:light blue",
        linestyle="dashed",
    )
    plt.plot(
        d2["times"],
        d2["metal_masses"][:, 1],
        color="xkcd:light red",
        linestyle="dashed",
    )
    plt.plot(
        d2["times"],
        d2["dust_masses"],
        color="xkcd:baby shit brown",
        linestyle="dashed",
    )


plt.suptitle("Reference Model")
plt.title("Component Mass Overview")
plt.ylabel("Component Mass (Msol)")
plt.xlabel("Time (Gyr)")
plt.yscale("log")
plt.legend(["gas", "stars", "metals", "oxygen", "dust"])
plt.savefig("plots/reference.eps")
plt.clf()
