import numpy as np

import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
from matplotlib.colors import to_rgb, rgb_to_hsv, hsv_to_rgb
from dustdevol.generic import (
    stellar_lifetimes,
    stellar_lifetimes_pchip,
    stellar_lifetimes_lin,
    stellar_lifetimes_nn,
    h_stellar_lifetimes,
    h_stellar_lifetimes_pchip,
    h_stellar_lifetimes_lin,
    h_stellar_lifetimes_nn,
)

metallicity = np.linspace(0.001, 0.04, 1000)
metallicity_val = np.linspace(0.2, 1.0, 1000)

value_dict = {}
for z, val in zip(metallicity, metallicity_val):
    value_dict[z] = val

masses = np.logspace(np.log10(0.8), np.log10(120), 1000)

models = [
    [stellar_lifetimes, h_stellar_lifetimes],
    [stellar_lifetimes_pchip, h_stellar_lifetimes_pchip],
    [stellar_lifetimes_lin, h_stellar_lifetimes_lin],
    [stellar_lifetimes_nn, h_stellar_lifetimes_nn],
]

model_names = [
    "Cubic",
    "PCHIP",
    "Linear",
    "NN",
]
colors = ["#0000ff", "#5500aa", "#aa0055", "#ff0000"]

gs = grid_spec.GridSpec(len(model_names), 1)
fig = plt.figure(figsize=(16, 9))


ax_objs = []
for i, name in enumerate(model_names):

    model = models[i]

    # creating new axes object
    ax_objs.append(fig.add_subplot(gs[i : i + 1, 0:]))

    # plotting the distribution
    for z in metallicity:
        col = rgb_to_hsv(to_rgb(colors[i]))
        col[1] = value_dict[z]
        ax_objs[-1].plot(
            np.log10(masses),
            np.log10(model[0]((z, masses))) + 2.75,
            color=hsv_to_rgb(col),
            alpha=0.2,
            lw=1,
        )
        col[2] /= 4
        ax_objs[-1].plot(
            np.log10(masses),
            np.log10(model[1]((z, masses))) + 2.75,
            color=hsv_to_rgb(col),
            alpha=0.2,
            lw=1,
        )

    # setting uniform x and y lims
    ax_objs[-1].set_xlim(np.log10(0.8), np.log10(120))
    ax_objs[-1].set_ylim(np.log10(0.001) + 2.75, np.log10(10000))

    # make background transparent
    rect = ax_objs[-1].patch
    rect.set_alpha(0)

    # remove borders, axis ticks, and labels
    ax_objs[-1].set_yticklabels([])
    ax_objs[-1].set_yticks([])

    if i == len(model_names) - 1:
        ax_objs[-1].set_xlabel("Progenitor Mass", fontsize=16)
    else:
        ax_objs[-1].set_xticklabels([])
        ax_objs[-1].set_xticks([])

    spines = ["top", "right", "left", "bottom"]
    for s in spines:
        ax_objs[-1].spines[s].set_visible(False)

    adj_country = name.replace(" ", "\n")
    ax_objs[-1].text(-0.12, 0, adj_country, fontsize=14, ha="right")

gs.update(hspace=-0.8)

fig.suptitle(
    "Stellar Lifetimes With Varying Models",
    fontsize=20,
)

plt.savefig("plots/lifetime_comparison.pdf", bbox_inches="tight")
