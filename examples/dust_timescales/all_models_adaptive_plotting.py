from numpy import load, log10, diff
import matplotlib.pyplot as plt

titles = ["IV", "V", "VI", "VII"]
models = ["Mattson", "bad_DeVis", "DeVis", "BEDE", "Asano"]
efficiencies = ["Mattson", "Bad DeVis", "DeVis", "BEDE", "Asano"]
timescales = ["Mattson", "Bad DeVis", "DeVis",
              "BEDE Diffuse", "BEDE Cloud", "Asano"]


for title in titles:
    for model in models:
        with load("outputs/Model_{}_{}_gg.npz".format(title, model)) as data:
            plt.figure(1)
            plt.plot(
                data["times"],
                data["gg_efficiency"],
            )

            plt.figure(2)
            if model == "BEDE":
                plt.plot(
                    data["times"],
                    data["gg_diffuse_timescale"],
                )
                plt.plot(
                    data["times"],
                    data["gg_cloud_timescale"],
                )
            else:
                plt.plot(
                    data["times"],
                    data["gg_timescale"],
                )

            plt.figure(3)
            plt.plot(
                data["gas_masses"] /
                    (data["gas_masses"] + data["star_masses"]),
                data["gg_efficiency"],
            )

            plt.figure(4)
            if model == "BEDE":
                plt.plot(
                    data["gas_masses"] /
                        (data["gas_masses"] + data["star_masses"]),
                    data["gg_diffuse_timescale"],
                )
                plt.plot(
                    data["gas_masses"] /
                        (data["gas_masses"] + data["star_masses"]),
                    data["gg_cloud_timescale"],
                )
            else:
                plt.plot(
                    data["gas_masses"] /
                        (data["gas_masses"] + data["star_masses"]),
                    data["gg_timescale"],
                )

            plt.figure(5)
            plt.plot(
                data["metal_masses"][:, 0] / data["gas_masses"][:, 0],
                data["gg_efficiency"],
            )

            plt.figure(6)
            if model == "BEDE":
                plt.plot(
                    data["metal_masses"][:, 0] / data["gas_masses"][:, 0],
                    data["gg_diffuse_timescale"],
                )
                plt.plot(
                    data["metal_masses"][:, 0] / data["gas_masses"][:, 0],
                    data["gg_cloud_timescale"],
                )
            else:
                plt.plot(
                    data["metal_masses"][:, 0] / data["gas_masses"][:, 0],
                    data["gg_timescale"],
                )

            plt.figure(7)
            plt.plot(
                12
                + log10(
                    (data["metal_masses"][:, 1] / 16) /
                    (data["gas_masses"][:, 0] / 1.36)
                ),
                data["gg_efficiency"],
            )

            plt.figure(8)
            if model == "BEDE":
                plt.plot(
                    12
                    + log10(
                        (data["metal_masses"][:, 1] / 16) /
                        (data["gas_masses"][:, 0] / 1.36)
                    ),
                    data["gg_diffuse_timescale"],
                )
                plt.plot(
                    12
                    + log10(
                        (data["metal_masses"][:, 1] / 16) /
                        (data["gas_masses"][:, 0] / 1.36)
                    ),
                    data["gg_cloud_timescale"],
                )
            else:
                plt.plot(
                    12
                    + log10(
                        (data["metal_masses"][:, 1] / 16) /
                        (data["gas_masses"][:, 0] / 1.36)
                    ),
                    data["gg_timescale"],
                )

            plt.figure(9)
            plt.plot(
                data["times"][:-1],
                diff(data["times"]),
            )

    plt.figure(1)
    plt.suptitle("Model " + title)
    plt.title("Dust Grow Efficiency Vs. Time")
    plt.ylabel("Efficiency (Gyr Msol / Msol)")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.legend(efficiencies)
    plt.savefig("plots/Model_" + title + "_gg_eff.eps")
    plt.clf()

    plt.figure(2)
    plt.suptitle("Model " + title)
    plt.title("Dust Growth Timescale Vs. Time")
    plt.ylabel("Timescale (Gyr Msol / Msol)")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.legend(timescales)
    plt.savefig("plots/Model_" + title + "_gg_time.eps")
    plt.clf()

    plt.figure(3)
    plt.suptitle("Model " + title)
    plt.title("Dust Grow Efficiency Vs. Gas Fraction")
    plt.ylabel("Efficiency (Gyr Msol / Msol)")
    plt.xlabel("Gas Fraction (Mg / (Mg + M*))")
    plt.yscale("log")
    plt.gca().xaxis.set_inverted(True)
    plt.legend(efficiencies)
    plt.savefig("plots/Model_" + title + "_gg_eff_gas_frac.eps")
    plt.clf()

    plt.figure(4)
    plt.suptitle("Model " + title)
    plt.title("Dust Growth Timescale Vs. Gas Fraction")
    plt.ylabel("Timescale (Gyr Msol / Msol)")
    plt.xlabel("Gas Fraction (Mg / (Mg + M*))")
    plt.yscale("log")
    plt.gca().xaxis.set_inverted(True)
    plt.legend(timescales)
    plt.savefig("plots/Model_" + title + "_gg_time_gas_frac.eps")
    plt.clf()

    plt.figure(5)
    plt.suptitle("Model " + title)
    plt.title("Dust Grow Efficiency Vs. Metallicity")
    plt.ylabel("Efficiency (Gyr Msol / Msol)")
    plt.xlabel("Metallicity (Mz / Mg)")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend(efficiencies)
    plt.savefig("plots/Model_" + title + "_gg_eff_metallicity.eps")
    plt.clf()

    plt.figure(6)
    plt.suptitle("Model " + title)
    plt.title("Dust Growth Timescale Vs. Metallicity")
    plt.ylabel("Timescale (Gyr Msol / Msol)")
    plt.xlabel("Metallicity (Mz / Mg)")
    plt.yscale("log")
    plt.xscale("log")
    plt.legend(timescales)
    plt.savefig("plots/Model_" + title + "_gg_time_metallicity.eps")
    plt.clf()

    plt.figure(7)
    plt.suptitle("Model " + title)
    plt.title("Dust Grow Efficiency Vs. 12 + log(O/H)")
    plt.ylabel("Efficiency (Gyr Msol / Msol)")
    plt.xlabel("12 + log(O/H)")
    plt.yscale("log")
    plt.legend(efficiencies)
    plt.savefig("plots/Model_" + title + "_gg_eff_logOH.eps")
    plt.clf()

    plt.figure(8)
    plt.suptitle("Model " + title)
    plt.title("Dust Growth Timescale Vs. 12 + log(O/H)")
    plt.ylabel("Timescale (Gyr Msol / Msol)")
    plt.xlabel("12 + log(O/H)")
    plt.yscale("log")
    plt.legend(timescales)
    plt.savefig("plots/Model_" + title + "_gg_time_logOH.eps")
    plt.clf()

    plt.figure(9)
    plt.suptitle("Model " + title)
    plt.title("Timestep Vs. Time")
    plt.ylabel("Timestep (Gyr)")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.legend(efficiencies)
    plt.savefig("plots/Model_" + title + "_gg_timesteps.eps")
    plt.clf()
