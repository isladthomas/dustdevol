from numpy import load, maximum, exp, minimum, array, pi, where, nan
import matplotlib.pyplot as plt
from dustdevol.generic import z_at_t
from dustdevol.BEDE2.gas import mass_loading
from models import model_grid, model_parameters, dict_configs

for params in dict_configs(model_parameters):
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
            + "_".join([str(x) for x in params.values()])
            + ".npz",
            allow_pickle=True,
        ) as d:
            model_times.append(d["times"])
            model_gg_time.append(d["gg_efficiency"])
            model_cloud_time.append(d["gg_cloud_timescale"])
            model_diffuse_time.append(d["gg_diffuse_timescale"])
            model_dd_time.append(
                (
                    ((1 - 0.5) * (1 - 0.1) / d["dd_frag_timescale"])
                    + ((1 - 0.5) / d["dd_des_timescale"])
                )
                ** (-1)
            )
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
        "plots/" + "_".join([str(x) for x in params.values()]) + "_gg_timescale.pdf",
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
        "plots/" + "_".join([str(x) for x in params.values()]) + "_dd_timescale.pdf",
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
        "plots/" + "_".join([str(x) for x in params.values()]) + "_cloud_timescale.pdf",
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
        "plots/"
        + "_".join([str(x) for x in params.values()])
        + "_diffuse_timescale.pdf",
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
        "plots/" + "_".join([str(x) for x in params.values()]) + "_des_timescale.pdf",
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
        "plots/" + "_".join([str(x) for x in params.values()]) + "_frag_timescale.pdf",
        bbox_inches="tight",
    )
    plt.clf()
