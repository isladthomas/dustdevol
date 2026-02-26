from numpy import load, maximum, exp, minimum
import matplotlib.pyplot as plt
from dustdevol.generic import z_at_t
from dustdevol.BEDE2.gas import mass_loading

with load("outputs/reference.npz") as data:
    plt.plot(
        data["times"],
        data["gas_masses"],
    )
    plt.plot(
        data["times"],
        data["star_masses"],
    )
    plt.plot(
        data["times"],
        data["metal_masses"][:, 0],
    )
    plt.plot(
        data["times"],
        data["dust_masses"],
    )
    plt.plot(
        data["times"],
        data["sfr"],
    )

plt.suptitle("Reference")
plt.title("Component Mass Overview")
plt.ylabel("Component Derivative (Msol/Gyr)")
plt.xlabel("Time (Gyr)")
plt.yscale("log")
plt.legend(["gas", "stars", "metals", "dust", "sfr"])
plt.savefig("plots/reference.eps")
plt.clf()
