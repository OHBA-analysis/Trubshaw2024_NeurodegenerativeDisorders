"""Plot results.

"""

import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import os

from osl_dynamics.analysis import power
from osl_dynamics.utils import plotting
import matplotlib.pyplot as plt

plotting.set_style({
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
})

comparisons = ["ALS-HC", "Asymp_C9-HC", "Asymp_SOD-HC", "Asymp_C9-Asymp_SOD"]
colors = [(0.01, 0.65, 0.08, 1), (1, 1, 1, 0)]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=256)
for i, name in enumerate(comparisons):
    cope = np.load(f"data/contrast_{i}.npy")
    pvalues = np.load(f"data/contrast_{i}_pvalues.npy")

    power.save(
        cope,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "cmap": "bwr",
            "bg_on_data": 1,
            "darkness": 1,
            "alpha": 0.7,
            "vmin": -0.001,
            "vmax": 0.001,
            "views": ["lateral"],
            "symmetric_cbar": True,
        },
        filename=f"plots/{name}_.png",
    )

    power.save(
        pvalues,
        mask_file="MNI152_T1_8mm_brain.nii.gz",
        parcellation_file="Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii.gz",
        plot_kwargs={
            "cmap": cmap,
            "bg_on_data": 1,
            "darkness": 1,
            "alpha": 0.7,
            "views": ["lateral"],
            "vmin": 0,
            "vmax": 0.1,
        },
        filename=f"plots/{name}_pvalues_.png",
    )

    plotting.close()


cols = len(comparisons)
for mode in range(6):
    fig, ax = plt.subplots(nrows=2, ncols=cols, figsize=(cols,2))
    for i, name in enumerate(comparisons):
        im1 = plt.imread(f"plots/{name}_{mode}.png")
        im2 = plt.imread(f"plots/{name}_pvalues_{mode}.png")
        ax[0, i].imshow(im1)
        ax[1, i].imshow(im2)
        #ax[0, i].set_title(name.upper(),fontsize=5)
        ax[0, i].axis("off")
        ax[1, i].axis("off")

    filename = f"plots/compiled_plots/power_{mode}.png"
    print("Saving", filename)
    plt.tight_layout()
    plt.savefig(filename, dpi=1000)
