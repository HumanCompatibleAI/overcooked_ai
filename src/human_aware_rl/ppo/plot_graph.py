import os

import matplotlib.font_manager
import matplotlib.pyplot as plt
import numpy as np
from evaluate import eval_models
from matplotlib.patches import Patch


# importing from utils causes werid dependency conflicts. Copying here
def set_style(font_scale=1.6):
    import matplotlib
    import seaborn

    seaborn.set(font="serif", font_scale=font_scale)
    # Make the background white, and specify the specific font family
    seaborn.set_style(
        "white",
        {
            "font.family": "serif",
            "font.weight": "normal",
            "font.serif": ["Times", "Palatino", "serif"],
            "axes.facecolor": "white",
            "lines.markeredgewidth": 1,
        },
    )
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rc("font", family="serif", serif=["Palatino"])


# each one is a len-5 dictionary, each value is a tuple of mean and se
PSP_PSP_0, hp_PSP_0, hp_PBC_0, hp_BC_0, bc_PBC_0 = eval_models(0)
_, hp_PSP_1, hp_PBC_1, hp_BC_1, bc_PBC_1 = eval_models(1)


def get_value(dic, pos):
    """
    The dictionary consists of layout:(mean, std), and we extract either the mean or the std based on its position
    """
    assert pos == 0 or pos == 1
    ls = []
    for key, values in dic.items():
        ls.append(values[pos])
    return ls


results_0 = [
    get_value(PSP_PSP_0, 0),
    get_value(hp_PSP_0, 0),
    get_value(hp_PBC_0, 0),
    get_value(hp_BC_0, 0),
    get_value(hp_PSP_1, 0),
    get_value(hp_PBC_1, 0),
    get_value(hp_BC_1, 0),
]
dotted_line = [get_value(bc_PBC_0, 0), get_value(bc_PBC_1, 0)]
stds = [
    get_value(PSP_PSP_0, 1),
    get_value(hp_PSP_0, 1),
    get_value(hp_PBC_0, 1),
    get_value(hp_BC_0, 1),
    get_value(hp_PSP_1, 1),
    get_value(hp_PBC_1, 1),
    get_value(hp_BC_1, 1),
]


hist_algos = [
    "PPO_SP+PPO_SP",
    "PPO_SP+HP",
    "PPO_BC+HP",
    "BC+HP",
    "HP+PPO_SP",
    "HP+PPO_BC",
    "HP+BC",
]
set_style()

fig, ax0 = plt.subplots(1, figsize=(18, 6))  # figsize=(20,6))

plt.rc("legend", fontsize=21)
plt.rc("axes", titlesize=25)
ax0.tick_params(axis="x", labelsize=18.5)
ax0.tick_params(axis="y", labelsize=18.5)

# there are 5 layouts
ind = np.arange(5)
width = 0.1
deltas = [-2.9, -1.5, -0.5, 0.5, 1.9, 2.9, 3.9]  # [-1, 0, 1, 2, 2.5, 3]
colors = ["#aeaeae", "#2d6777", "#F79646"]
# for each algo, total of 7
# in each loop, we plot the result for all 5 layouts for each algo
for i in range(len(hist_algos)):
    delta, algo = deltas[i], hist_algos[i]
    offset = ind + delta * width
    if i == 0:
        # this is the self-play vs self-play results, we don't want any color
        color = "none"
    else:
        color = colors[i % 3]
    if i == 0:
        ax0.bar(
            offset,
            results_0[i],
            width,
            color=color,
            edgecolor="gray",
            lw=1.0,
            zorder=0,
            label=algo,
            linestyle=":",
            yerr=stds[i],
        )
    elif 1 <= i <= 3:
        ax0.bar(
            offset,
            results_0[i],
            width,
            color=color,
            lw=1.0,
            zorder=0,
            label=algo,
            yerr=stds[i],
        )
    else:
        ax0.bar(
            offset,
            results_0[i],
            width,
            color=color,
            edgecolor="white",
            lw=1.0,
            zorder=0,
            hatch="/",
            yerr=stds[i],
        )
fst = True
for h_line in dotted_line:
    if fst:
        ax0.hlines(
            h_line[0],
            xmin=-0.4,
            xmax=0.4,
            colors="red",
            label="PPO_BC+BC",
            linestyle=":",
        )
        fst = False
    else:
        ax0.hlines(h_line[0], xmin=-0.4, xmax=0.4, colors="red", linestyle=":")
    ax0.hlines(h_line[1], xmin=0.6, xmax=1.4, colors="red", linestyle=":")
    ax0.hlines(h_line[2], xmin=1.6, xmax=2.4, colors="red", linestyle=":")
    ax0.hlines(h_line[3], xmin=2.6, xmax=3.45, colors="red", linestyle=":")
    ax0.hlines(h_line[4], xmin=3.6, xmax=4.4, colors="red", linestyle=":")
ax0.set_ylabel("Average reward per episode")
ax0.set_title("Performance with Human Proxy Models")

ax0.set_xticks(ind + width / 3)
ax0.set_xticklabels(
    (
        "Cramped Rm.",
        "Asymm. Adv.",
        "Coord. Ring",
        "Forced Coord.",
        "Counter Circ.",
    )
)

ax0.tick_params(axis="x", labelsize=18)
handles, labels = ax0.get_legend_handles_labels()
patch = Patch(
    facecolor="white",
    edgecolor="black",
    hatch="/",
    alpha=0.5,
    label="Switched start indices",
)
handles.append(patch)

# plot the legend
ax0.legend(handles=handles, loc="best")

ax0.set_ylim(0, 250)

plt.savefig("graph.jpg", format="jpg", bbox_inches="tight")
plt.show()
