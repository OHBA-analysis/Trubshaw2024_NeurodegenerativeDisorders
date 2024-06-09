"""Save source reconstructed data as numpy files.

"""

import os
import mne
import numpy as np
import pandas as pd

output_dir = f"../../data/src/npy"
os.makedirs(output_dir, exist_ok=True)

participants = pd.read_csv(f"../../demographics/demographics_als_dyn.csv")
subjects = participants["Subject"].values

# create subjects filelist
parc_paths = []
for subject in subjects:
    parc_paths.append(f"/home/mtrubshaw/Documents/ALS_dyn/data/src/{subject}/sflip_parc-raw.fif")

for path, subject in zip(parc_paths,subjects):
    raw = mne.io.read_raw_fif(path, verbose=False)
    raw.pick("misc")
    data = raw.get_data(reject_by_annotation="omit", verbose=False).T
    np.save(f"{output_dir}/{subject}.npy", data)
    print(subject)
