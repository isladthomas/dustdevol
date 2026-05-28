from numpy import load, maximum, linspace, exp, minimum, array, pi, where, nan, nanmax, stack, diff
import matplotlib.pyplot as plt
from dustdevol.generic import z_at_t
from dustdevol.BEDE2.gas import mass_loading
from models import model_grid, dict_configs
from scipy.ndimage import gaussian_filter1d
import matplotlib.ticker as mtick

model_times = []
model_gg_time = []
model_cloud_time = []
model_diffuse_time = []
model_dd_time = []
model_des_time = []
model_frag_time = []
model_tot_dust_time = []
legend = []
for model in dict_configs(model_grid):

    with load(
        "outputs/"
        + "_".join([x.__name__ for x in model.values()])
        + "_"
        + "_lifetime_updated.npz",
        allow_pickle=True,
    ) as d:
        model_times.append(d["times"])
        model_gg_time.append(d["gg_efficiency"])
        model_cloud_time.append(d["gg_cloud_timescale"])
        model_diffuse_time.append(d["gg_diffuse_timescale"])
        model_dd_time.append(d["dd_efficiency"])
#            (
#                ((1 - 0.5) * (1 - 0.1) / d["dd_frag_timescale"])
#                + ((1 - 0.5) / d["dd_des_timescale"])
#            )
#            ** (-1)
#        )
        model_des_time.append(d["dd_des_timescale"])
        model_frag_time.append(d["dd_frag_timescale"])
        model_tot_dust_time.append(d["dust_masses"][:, 0] / d["ddust_masses"][:, 0])
        legend.append(" ".join([x.__name__ for x in model.values()]))

for i in range(len(model_times)):
    plt.plot(model_times[i], model_gg_time[i][:, 0], linestyle=(i * pi, (3, 10)))
plt.plot(
    model_times[1],
    where(model_tot_dust_time[1] > 0, model_tot_dust_time[1], nan),
    color="xkcd:shit brown",
)
plt.plot(
    model_times[1],
    where(model_tot_dust_time[1] < 0, -model_tot_dust_time[1], nan),
    color="xkcd:baby shit brown",
    linestyle="dotted",
)

plt.suptitle("Grain Growth Timescales")
plt.title("Model Comparison")
plt.ylabel("GG Time (Gyr Msol / Msol)")
plt.xlabel("Time (Gyr)")
plt.yscale("log")
plt.legend(legend)
plt.savefig(
    "plots/gg_timescale.pdf",
    bbox_inches="tight",
)
plt.clf()

for i in range(len(model_times)):
    plt.plot(model_times[i], model_dd_time[i][:, 0], linestyle=(i * pi, (3, 10)))

plt.plot(
    model_times[1],
    where(model_tot_dust_time[1] < 0, -model_tot_dust_time[1], nan),
    color="xkcd:shit brown",
)
plt.plot(
    model_times[1],
    where(model_tot_dust_time[1] > 0, model_tot_dust_time[1], nan),
    color="xkcd:baby shit brown",
    linestyle="dotted",
)

plt.suptitle("Dust Destruction Timescales")
plt.title("Model Comparison")
plt.ylabel("DD Time (Gyr Msol / Msol)")
plt.xlabel("Time (Gyr)")
plt.yscale("log")
plt.legend(legend)
plt.savefig(
    "plots/dd_timescale.pdf",
    bbox_inches="tight",
)
plt.clf()

for i in range(len(model_times)):
    plt.plot(model_times[i], model_cloud_time[i][:, 0], linestyle=(i * pi, (3, 10)))

plt.suptitle("Cloud Grain Growth Timescales")
plt.title("Model Comparison")
plt.ylabel("GG Time (Gyr Msol / Msol)")
plt.xlabel("Time (Gyr)")
plt.yscale("log")
plt.legend(legend)
plt.savefig(
    "plots/cloud_timescale.pdf",
    bbox_inches="tight",
)
plt.clf()

for i in range(len(model_times)):
    plt.plot(
        model_times[i], model_diffuse_time[i][:, 0], linestyle=(i * pi, (3, 10))
    )

plt.suptitle("Diffuse Grain Growth Timescales")
plt.title("Model Comparison")
plt.ylabel("GG Time (Gyr Msol / Msol)")
plt.xlabel("Time (Gyr)")
plt.yscale("log")
plt.legend(legend)
plt.savefig(
    "plots/diffuse_timescale.pdf",
    bbox_inches="tight",
)
plt.clf()

for i in range(len(model_times)):
    plt.plot(model_times[i], model_des_time[i][:, 0], linestyle=(i * pi, (3, 10)))

plt.suptitle("SNe Dust Destruction Timescales")
plt.title("Model Comparison")
plt.ylabel("DD Time (Gyr Msol / Msol)")
plt.xlabel("Time (Gyr)")
plt.yscale("log")
plt.legend(legend)
plt.savefig(
    "plots/des_timescale.pdf",
    bbox_inches="tight",
)
plt.clf()

for i in range(len(model_times)):
    plt.plot(model_times[i], model_frag_time[i][:, 0], linestyle=(i * pi, (3, 10)))

plt.suptitle("Photofragmentation Timescales")
plt.title("Model Comparison")
plt.ylabel("DD Time (Gyr Msol / Msol)")
plt.xlabel("Time (Gyr)")
plt.yscale("log")
plt.legend(legend)
plt.savefig(
    "plots/frag_timescale.pdf",
    bbox_inches="tight",
)
plt.clf()

with load(
    "outputs/ref_lifetime_updated.npz",
    allow_pickle=True,
) as d1, load(
    "outputs/Ia_off_lifetime_updated.npz",
    allow_pickle=True,
) as d2, load(
    "outputs/bad_z_lifetime_updated.npz",
    allow_pickle=True,
) as d3:
    plt.plot(d1["times"], d1["sfr"])
    plt.plot(d3["times"], d3["sfr"])

    plt.suptitle("SFR Comparisons")
    plt.title("Redshift On vs. Off")
    plt.ylabel("SFR (Msol/Gyr)")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.legend(["redshift on", "redshift off"])
    plt.savefig(
        "plots/redshift.pdf",
        bbox_inches="tight",
    )
    plt.clf()

    times = linspace(0, 13.8, 1000)
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
    nitro_data = gaussian_filter1d(
        (d2["metal_func"].item()(times) - d1["metal_func"].item()(times))
        / d1["metal_func"].item()(times)
        * 100,
        5,
        axis=0,
    )[:, 2]
    carb_data = gaussian_filter1d(
        (d2["metal_func"].item()(times) - d1["metal_func"].item()(times))
        / d1["metal_func"].item()(times)
        * 100,
        5,
        axis=0,
    )[:, 3]
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
        color="xkcd:orange",
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
        where(nitro_data > 0, nitro_data, nan),
        color="xkcd:blue",
        linestyle="solid",
    )
    plt.plot(
        times,
        where(carb_data > 0, carb_data, nan),
        color="xkcd:black",
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
        color="xkcd:light orange",
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
        where(nitro_data <= 0, -nitro_data, nan),
        color="xkcd:light blue",
        linestyle="dashed",
    )
    plt.plot(
        times,
        where(carb_data <= 0, -carb_data, nan),
        color="xkcd:grey",
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
        color="xkcd:light orange",
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
        (
            abs(d2["metal_func"].item()(times) - d1["metal_func"].item()(times))
            / d1["metal_func"].item()(times)
        )[:, 2]
        * 100,
        color="xkcd:light blue",
        alpha=0.2,
    )
    plt.plot(
        times,
        (
            abs(d2["metal_func"].item()(times) - d1["metal_func"].item()(times))
            / d1["metal_func"].item()(times)
        )[:, 3]
        * 100,
        color="xkcd:grey",
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

    plt.title("Models with and without SNIa")
    plt.ylabel("Percent Difference")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.ylim(
        0.01,
        nanmax(abs(stack((gas_data, star_data, metal_data, oxygen_data, nitro_data, carb_data, dust_data))))
        * 10 ** (0.1),
    )
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(100, 3))
    plt.legend(["gas", "stars", "metals", "oxygen", "nitrogen", "carbon", "dust"])
    plt.savefig("plots/Ia_comp.pdf", bbox_inches="tight")
    plt.clf()

    plt.plot(d1["times"][:-1], diff(d1["times"]))

    plt.title("Adaptive Model Timesteps")
    plt.ylabel("Timestep (Gyr)")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.savefig("plots/timesteps.pdf", bbox_inches="tight")
    plt.clf()
