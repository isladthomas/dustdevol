from numpy import load, maximum, exp, minimum, linspace, where, nan, stack, nanmax
import matplotlib.pyplot as plt
from dustdevol.generic import z_at_t
from dustdevol.BEDE2.gas import mass_loading
from scipy.ndimage import gaussian_filter1d
import matplotlib.ticker as mtick
from models import model_names
import logging

times = linspace(0, 13.8, 1000)

logging.captureWarnings(True)
logging.basicConfig(filename="Plot_Warnings.log", level=logging.WARNING)

for name in model_names[:, 0, :, :].flatten():
    suptitle = "Cubic vs NN Stellar Lifetime Comparison"
    title = (
        name["death_point"].capitalize()
        + " Burning, "
        + name["birth_death_calc"].capitalize()
        + " Lifetime Calc, "
        + "SNIa "
        + name["type_Ia"].capitalize()
    )
    ylabel = "Percent Difference"
    plot_name = (
        name["death_point"]
        + "_compnn_"
        + name["type_Ia"]
        + "_"
        + name["birth_death_calc"]
    )
    file1 = (
        "outputs/"
        + name["death_point"]
        + "_cube_"
        + name["type_Ia"]
        + "_"
        + name["birth_death_calc"]
        + ".npz"
    )
    file2 = (
        "outputs/"
        + name["death_point"]
        + "_nn_"
        + name["type_Ia"]
        + "_"
        + name["birth_death_calc"]
        + ".npz"
    )

    # note: solid line means second model overpredicts, dashed means underpredicts
    with load(file1, allow_pickle=True) as d1, load(file2, allow_pickle=True) as d2:

        gas_data = gaussian_filter1d(
            (d2["gas_func"].item()(times) - d1["gas_func"].item()(times))
            / d1["gas_func"].item()(times)
            * 100,
            5,
            axis=0,
        )[:, 0]
        star_data = gaussian_filter1d(
            (d2["star_func"].item()(times) - d1["star_func"].item()(times))
            / d1["star_func"].item()(times)
            * 100,
            5,
            axis=0,
        )[:, 0]
        metal_data = gaussian_filter1d(
            (d2["metal_func"].item()(times) - d1["metal_func"].item()(times))
            / d1["metal_func"].item()(times)
            * 100,
            5,
            axis=0,
        )[:, 0]
        oxygen_data = gaussian_filter1d(
            (d2["metal_func"].item()(times) - d1["metal_func"].item()(times))
            / d1["metal_func"].item()(times)
            * 100,
            5,
            axis=0,
        )[:, 1]
        dust_data = gaussian_filter1d(
            (d2["dust_func"].item()(times) - d1["dust_func"].item()(times))
            / d1["dust_func"].item()(times)
            * 100,
            5,
            axis=0,
        )[:, 0]

        plt.plot(
            times,
            where(gas_data > 0, gas_data, nan),
            color="xkcd:green",
            linestyle="solid",
        )
        plt.plot(
            times,
            where(star_data > 0, star_data, nan),
            color="xkcd:purple",
            linestyle="solid",
        )
        plt.plot(
            times,
            where(metal_data > 0, metal_data, nan),
            color="xkcd:blue",
            linestyle="solid",
        )
        plt.plot(
            times,
            where(oxygen_data > 0, oxygen_data, nan),
            color="xkcd:red",
            linestyle="solid",
        )
        plt.plot(
            times,
            where(dust_data > 0, dust_data, nan),
            color="xkcd:shit brown",
            linestyle="solid",
        )

        plt.plot(
            times,
            where(gas_data <= 0, -gas_data, nan),
            color="xkcd:light green",
            linestyle="dashed",
        )
        plt.plot(
            times,
            where(star_data <= 0, -star_data, nan),
            color="xkcd:light purple",
            linestyle="dashed",
        )
        plt.plot(
            times,
            where(metal_data <= 0, -metal_data, nan),
            color="xkcd:light blue",
            linestyle="dashed",
        )
        plt.plot(
            times,
            where(oxygen_data <= 0, -oxygen_data, nan),
            color="xkcd:light red",
            linestyle="dashed",
        )
        plt.plot(
            times,
            where(dust_data <= 0, -dust_data, nan),
            color="xkcd:baby shit brown",
            linestyle="dashed",
        )

        plt.plot(
            times,
            abs(d2["gas_func"].item()(times) - d1["gas_func"].item()(times))
            / d1["gas_func"].item()(times)
            * 100,
            color="xkcd:light green",
            alpha=0.2,
        )
        plt.plot(
            times,
            abs(d2["star_func"].item()(times) - d1["star_func"].item()(times))
            / d1["star_func"].item()(times)
            * 100,
            color="xkcd:light purple",
            alpha=0.2,
        )
        plt.plot(
            times,
            (
                abs(d2["metal_func"].item()(times) - d1["metal_func"].item()(times))
                / d1["metal_func"].item()(times)
            )[:, 0]
            * 100,
            color="xkcd:light blue",
            alpha=0.2,
        )
        plt.plot(
            times,
            (
                abs(d2["metal_func"].item()(times) - d1["metal_func"].item()(times))
                / d1["metal_func"].item()(times)
            )[:, 1]
            * 100,
            color="xkcd:light red",
            alpha=0.2,
        )
        plt.plot(
            times,
            abs(d2["dust_func"].item()(times) - d1["dust_func"].item()(times))
            / d1["dust_func"].item()(times)
            * 100,
            color="xkcd:baby shit brown",
            alpha=0.2,
        )

    plt.suptitle(suptitle)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.ylim(
        0.0001,
        nanmax(abs(stack((gas_data, star_data, metal_data, oxygen_data, dust_data))))
        * 10 ** (0.1),
    )
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(100, 3))
    plt.legend(["gas", "stars", "metals", "oxygen", "dust"])
    plt.savefig("plots/" + plot_name + ".pdf", bbox_inches="tight")
    plt.clf()
