"""Script to extract aperiodic exponents from psds and to de-fooof psds
and calculate Center of Energy for each parcel and plot group average surface plots.

Center of Energy reference: https://movementdisorders.onlinelibrary.wiley.com/doi/full/10.1002/mds.29378
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fooof import FOOOF
from fooof.plts.spectra import plot_spectrum
from fooof.utils import trim_spectrum, interpolate_spectrum

from osl_dynamics.analysis import power

# -----------------------------------
# FOOOF

# load freqs and psds
freqs1 = np.load(f"../../data/static/f.npy")
psd = np.load("../../data/static/p.npy")

# extract defooofed frequencies
fm = FOOOF(min_peak_height=0.05, verbose=False)
fm.fit(freqs1, psd[0, 0, :], freq_range=[1, 70])  # only fit between 1Hz and 70Hz
defooofed_f = fm.freqs

# create empty variables
defooofed_spectrum = np.zeros((psd.shape[0], psd.shape[1], len(defooofed_f)))
aperiodic_exponents = np.zeros((len(psd), len(psd[0])))

# fit fooof to every psd, returns de-fooofed spectra and aperiodic exponents
for r in range(0, len(psd[0])):
    for i in range(len(psd)):
        # get this spectrum's psd
        powers = psd[i, r, :]

        # Interpolate 30Hz and 50Hz line noise
        interp_range = [[48, 52]]
        freqs_int, powers_int = interpolate_spectrum(freqs1, powers, interp_range)

        # Initialize power spectrum model objects and fit the power spectra
        fm = FOOOF(min_peak_height=0.05, verbose=False)
        fm.fit(
            freqs_int, powers_int, freq_range=[1, 70]
        )  # only fit between 1Hz and 70Hz

        # extract aperiodic exponents
        aperiodic_exponent = fm.aperiodic_params_[1]

        # extract the full model and aperiodic then calculate difference
        aperiodic = fm.get_model("aperiodic", space="linear")
        full = fm.get_model("full", space="linear")
        diff = full - aperiodic

        # #return de-fooofed spectrum
        defooofed_spectrum[i, r, :] = diff
        aperiodic_exponents[i, r] = aperiodic_exponent

        # Plots fooof spectrum
        # fm.plot(plot_aperiodic=True, plot_peaks='line-shade-outline', plt_log=False)

    print(r, "/", len(psd[0]))

np.save(f"../../data/static/defooofed_f.npy", defooofed_f)
np.save("../../data/static/defooofed_psd.npy", defooofed_spectrum)
np.save("../../data/static/aperiodic_exponents.npy", aperiodic_exponents)

# ------------------------------------------------
# Centre of energy

# Load data
demographics = pd.read_csv("../../demographics/demographics_als_dyn.csv")

f = np.load("../../data/static/defooofed_f.npy")
psd = np.load("../../data/static/defooofed_psd.npy")  # Here we can also load 1/f Subtracted spectra

# Remove data lower than 4Hz
f_in = np.logical_and(f >= 4, f <= 30)  # we might need to discuss this at some point
f = f[f_in]
psd = psd[:, :, f_in]
psd = psd + 0.0000000000000001

# Get Center of Energy Per Participant
CoE = np.sum(psd * f[np.newaxis, np.newaxis], axis=2) / np.sum(psd, axis=2)

np.save(f"../../data/static/coe.npy", CoE)
