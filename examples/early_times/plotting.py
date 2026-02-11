from numpy import load, log10, diff
import matplotlib.pyplot as plt

titles = ["I", "II", "III", "IV", "V", "VI", "VII"]
accuracies = ["1e0", "1e-3", "1e-6", "1e-9", "1e-12", "1e-15", "1e-18"]


for title in titles:
    for accuracy in accuracies:
        with load("outputs/Model_{}_{}_accuracy.npz".format(title, accuracy)) as data:
            plt.figure(1)
            plt.plot(
                data["times"],
                data["dust_masses"],
            )

            plt.figure(2)
            plt.plot(
                data["times"],
                data["metal_masses"][:, 0] / data["gas_masses"][:, 0],
            )

            plt.figure(3)
            plt.plot(
                data["times"],
                12
                + log10(
                    (data["metal_masses"][:, 1] / 16)
                    / (data["gas_masses"][:, 0] / 1.36)
                ),
            )

            plt.figure(4)
            plt.plot(
                data["times"][:-1],
                diff(data["times"]),
            )

    plt.figure(1)
    plt.suptitle("Model " + title)
    plt.title("Dust Mass Vs. Time")
    plt.ylabel("Dust Mass (Msol)")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.legend(accuracies)
    plt.savefig("plots/Model_" + title + "_dust.eps")
    plt.clf()

    plt.figure(2)
    plt.suptitle("Model " + title)
    plt.title("Metallicity Vs. Time")
    plt.ylabel("Metallicity (Mz / Mg)")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.legend(accuracies)
    plt.savefig("plots/Model_" + title + "_metallicity.eps")
    plt.clf()

    plt.figure(3)
    plt.suptitle("Model " + title)
    plt.title("12 + log(O/H) Vs. Time")
    plt.ylabel("12 + log(O/H)")
    plt.xlabel("Time (Gyr)")
    plt.legend(accuracies)
    plt.savefig("plots/Model_" + title + "_logOH.eps")
    plt.clf()

    plt.figure(4)
    plt.suptitle("Model " + title)
    plt.title("Timesteps Vs. Time")
    plt.ylabel("Timestep (Gyr)")
    plt.xlabel("Time (Gyr)")
    plt.yscale("log")
    plt.legend(accuracies)
    plt.savefig("plots/Model_" + title + "_timesteps.eps")
    plt.clf()
