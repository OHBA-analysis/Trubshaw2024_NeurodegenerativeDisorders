"""Calculate static (time-averaged) quantities.

"""

import os
import numpy as np
import pandas as pd

from osl_dynamics.data import Data
from osl_dynamics.analysis import static, power
from osl_dynamics.utils import plotting


static_dir = f"data/static"
os.makedirs(static_dir, exist_ok=True)

participants = pd.read_csv(f"../../demographics/demographics_als_dyn.csv")
subjects = participants["Subject"].values

# create subjects filelist
parc_paths = []
for subject in subjects:
    parc_paths.append(f"/home/mtrubshaw/Documents/ALS_dyn/data/src/{subject}/sflip_parc-raw.fif")

# load npy files
#combined_data = np.array([np.load(subject) for subject in subjects])
#combined_data_ls = combined_data.tolist()

data = Data(parc_paths, picks="misc", reject_by_annotation="omit", n_jobs=6)

# Calculate static power spectra
ts = data.time_series()
f, p = static.welch_spectra(
    data=ts,
    window_length=500,
    sampling_frequency=250,
    standardize=True,
    n_jobs=6,
)
np.save(f"../../data/static/p.npy", p)
np.save(f"../../data/static/f.npy", f)

n_subjects = len(subjects)
n_parcels = 52

# Calculate AEC for each frequency band - needs ', sampling_frequency=250' in Data()
data.set_sampling_frequency(250)

low_freqs = [1, 4, 7, 13, 30, 52]
high_freqs = [4, 7, 13, 30, 48, 80]

aec = []
for low_freq, high_freq in zip(low_freqs, high_freqs):
    methods = {
        "filter": {"low_freq": low_freq, "high_freq": high_freq, "use_raw": True},
        "amplitude_envelope": {},
        "standardize": {},
    }
    data.prepare(methods)
    d = data.time_series()
    fc = static.functional_connectivity(d)
    aec.append(fc)

np.save("../../data/static/aec.npy", aec)
