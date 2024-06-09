"""Signal-space projection (SSP) denoising.

"""

# Authors: Mats van Es <mats.vanes@psych.ox.ac.uk>
#          Oliver Kohl <oliver.kohl@psych.ox.ac.uk>
#          Chetan Gohil <chetan.gohil@psych.ox.ac.uk>

import os
import mne
import matplotlib.pyplot as plt
import pandas as pd

from osl import preprocessing


# Directories
preproc_dir = "/home/mtrubshaw/Documents/ALS_dyn/data/preproc"
ssp_preproc_dir = "/home/mtrubshaw/Documents/ALS_dyn/data/preproc_ssp"
report_dir = "/home/mtrubshaw/Documents/ALS_dyn/data/preproc_ssp/report"

os.makedirs(ssp_preproc_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

participants = pd.read_csv("/home/mtrubshaw/Documents/ALS_dyn/data/demographics/complete_demo_dyn_als.csv")


# 1 ---------------------------------------

# all subjects
subjects = participants["Subject"].values

ssp_preproc_files = []
preproc_files = []
for subject in subjects:
    preproc_files.append(f"{preproc_dir}/{subject}/{subject}_preproc_raw.fif")
    ssp_preproc_files.append(f"{ssp_preproc_dir}/{subject}/{subject}_preproc_raw.fif")



for index in range(len(preproc_files)):
    subject = subjects[index]
    preproc_file = preproc_files[index]
    output_raw_file = ssp_preproc_files[index]

    # Make output directory
    os.makedirs(os.path.dirname(output_raw_file), exist_ok=True)

    # Load preprocessed fif and ICA
    dataset = preprocessing.read_dataset(preproc_file, preload=True)
    raw = dataset["raw"]

    # Only keep MEG, ECG, EOG, EMG
    raw = raw.pick_types(meg=True, ecg=True, eog=True, emg=True)

    # Create a Raw object without any channels marked as bad
    raw_no_bad_channels = raw.copy()
    raw_no_bad_channels.load_bad_channels()

    #  Calculate SSP using ECG
    n_proj = 1
    ecg_epochs = mne.preprocessing.create_ecg_epochs(
        raw_no_bad_channels, picks="all"
    ).average(picks="all")
    ecg_projs, events = mne.preprocessing.compute_proj_ecg(
        raw_no_bad_channels,
        n_grad=n_proj,
        n_mag=n_proj,
        n_eeg=0,
        no_proj=True,
        reject=None,
        n_jobs=6,
    )

    # Add ECG SSPs to Raw object
    raw_ssp = raw.copy()
    raw_ssp.add_proj(ecg_projs.copy())

    # Calculate SSP using EOG
    n_proj = 1
    eog_epochs = mne.preprocessing.create_eog_epochs(
        raw_no_bad_channels, picks="all"
    ).average()
    eog_projs, events = mne.preprocessing.compute_proj_eog(
        raw_no_bad_channels,
        n_grad=n_proj,
        n_mag=n_proj,
        n_eeg=0,
        no_proj=True,
        reject=None,
        n_jobs=6,
    )

    # Add EOG SSPs to Raw object
    raw_ssp.add_proj(eog_projs.copy())

    # Apply SSPs
    raw_ssp.apply_proj()

    # Plot power spectrum of cleaned data
    raw_ssp.plot_psd(fmax=45, n_fft=int(raw.info["sfreq"] * 4))
    plt.savefig(f"{report_dir}/psd_{subject}.png", bbox_inches="tight")
    plt.close()

    if len(ecg_projs) > 0:
        fig = mne.viz.plot_projs_joint(ecg_projs, ecg_epochs, show=False)
        plt.savefig(f"{report_dir}/proj_ecg_{subject}.png", bbox_inches="tight")
        plt.close()

    if len(eog_projs) > 0:
        fig = mne.viz.plot_projs_joint(eog_projs, eog_epochs, show=False)
        plt.savefig(f"{report_dir}/proj_eog_{subject}.png", bbox_inches="tight")
        plt.close()

    # Save cleaned data
    raw_ssp.save(output_raw_file, overwrite=True)

#-----------------------------------------

#subect which need 4 ECG SSPs
subjects = ["sub-0175","sub-0241","sub-0244","sub-0245","sub-0246"]

ssp_preproc_files = []
preproc_files = []
for subject in subjects:
    preproc_files.append(f"{preproc_dir}/{subject}/{subject}_preproc_raw.fif")
    ssp_preproc_files.append(f"{ssp_preproc_dir}/{subject}/{subject}_preproc_raw.fif")

for index in range(len(preproc_files)):
    subject = subjects[index]
    preproc_file = preproc_files[index]
    output_raw_file = ssp_preproc_files[index]

    # Make output directory
    os.makedirs(os.path.dirname(output_raw_file), exist_ok=True)

    # Load preprocessed fif and ICA
    dataset = preprocessing.read_dataset(preproc_file, preload=True)
    raw = dataset["raw"]

    # Only keep MEG, ECG, EOG, EMG
    raw = raw.pick_types(meg=True, ecg=True, eog=True, emg=True)

    # Create a Raw object without any channels marked as bad
    raw_no_bad_channels = raw.copy()
    raw_no_bad_channels.load_bad_channels()

    #  Calculate SSP using ECG
    n_proj = 4
    ecg_epochs = mne.preprocessing.create_ecg_epochs(
        raw_no_bad_channels, picks="all"
    ).average(picks="all")
    ecg_projs, events = mne.preprocessing.compute_proj_ecg(
        raw_no_bad_channels,
        n_grad=n_proj,
        n_mag=n_proj,
        n_eeg=0,
        no_proj=True,
        reject=None,
        n_jobs=6,
    )

    # Add ECG SSPs to Raw object
    raw_ssp = raw.copy()
    raw_ssp.add_proj(ecg_projs.copy())

    # Calculate SSP using EOG
    n_proj = 1
    eog_epochs = mne.preprocessing.create_eog_epochs(
        raw_no_bad_channels, picks="all"
    ).average()
    eog_projs, events = mne.preprocessing.compute_proj_eog(
        raw_no_bad_channels,
        n_grad=n_proj,
        n_mag=n_proj,
        n_eeg=0,
        no_proj=True,
        reject=None,
        n_jobs=6,
    )

    # Add EOG SSPs to Raw object
    raw_ssp.add_proj(eog_projs.copy())

    # Apply SSPs
    raw_ssp.apply_proj()

    # Plot power spectrum of cleaned data
    raw_ssp.plot_psd(fmax=45, n_fft=int(raw.info["sfreq"] * 4))
    plt.savefig(f"{report_dir}/psd_{subject}.png", bbox_inches="tight")
    plt.close()

    if len(ecg_projs) > 0:
        fig = mne.viz.plot_projs_joint(ecg_projs, ecg_epochs, show=False)
        plt.savefig(f"{report_dir}/proj_ecg_{subject}.png", bbox_inches="tight")
        plt.close()

    if len(eog_projs) > 0:
        fig = mne.viz.plot_projs_joint(eog_projs, eog_epochs, show=False)
        plt.savefig(f"{report_dir}/proj_eog_{subject}.png", bbox_inches="tight")
        plt.close()

    # Save cleaned data
    raw_ssp.save(output_raw_file, overwrite=True)


# 3 ---------------------------------------

# subjects which need 3 ECG SSPs
subjects = ["sub-0043","sub-0087","sub-0116","sub-0122","sub-0145",
             "sub-0152","sub-0175","sub-0193","sub-0206","sub-0230",
             "sub-0241","sub-0244","sub-0245","sub-0246"]

ssp_preproc_files = []
preproc_files = []
for subject in subjects:
    preproc_files.append(f"{preproc_dir}/{subject}/{subject}_preproc_raw.fif")
    ssp_preproc_files.append(f"{ssp_preproc_dir}/{subject}/{subject}_preproc_raw.fif")

for index in range(len(preproc_files)):
    subject = subjects[index]
    preproc_file = preproc_files[index]
    output_raw_file = ssp_preproc_files[index]

    # Make output directory
    os.makedirs(os.path.dirname(output_raw_file), exist_ok=True)

    # Load preprocessed fif and ICA
    dataset = preprocessing.read_dataset(preproc_file, preload=True)
    raw = dataset["raw"]

    # Only keep MEG, ECG, EOG, EMG
    raw = raw.pick_types(meg=True, ecg=True, eog=True, emg=True)

    # Create a Raw object without any channels marked as bad
    raw_no_bad_channels = raw.copy()
    raw_no_bad_channels.load_bad_channels()

    #  Calculate SSP using ECG
    n_proj = 3
    ecg_epochs = mne.preprocessing.create_ecg_epochs(
        raw_no_bad_channels, picks="all"
    ).average(picks="all")
    ecg_projs, events = mne.preprocessing.compute_proj_ecg(
        raw_no_bad_channels,
        n_grad=n_proj,
        n_mag=n_proj,
        n_eeg=0,
        no_proj=True,
        reject=None,
        n_jobs=6,
    )

    # Add ECG SSPs to Raw object
    raw_ssp = raw.copy()
    raw_ssp.add_proj(ecg_projs.copy())

    # Calculate SSP using EOG
    n_proj = 1
    eog_epochs = mne.preprocessing.create_eog_epochs(
        raw_no_bad_channels, picks="all"
    ).average()
    eog_projs, events = mne.preprocessing.compute_proj_eog(
        raw_no_bad_channels,
        n_grad=n_proj,
        n_mag=n_proj,
        n_eeg=0,
        no_proj=True,
        reject=None,
        n_jobs=6,
    )

    # Add EOG SSPs to Raw object
    raw_ssp.add_proj(eog_projs.copy())

    # Apply SSPs
    raw_ssp.apply_proj()

    # Plot power spectrum of cleaned data
    raw_ssp.plot_psd(fmax=45, n_fft=int(raw.info["sfreq"] * 4))
    plt.savefig(f"{report_dir}/psd_{subject}.png", bbox_inches="tight")
    plt.close()

    if len(ecg_projs) > 0:
        fig = mne.viz.plot_projs_joint(ecg_projs, ecg_epochs, show=False)
        plt.savefig(f"{report_dir}/proj_ecg_{subject}.png", bbox_inches="tight")
        plt.close()

    if len(eog_projs) > 0:
        fig = mne.viz.plot_projs_joint(eog_projs, eog_epochs, show=False)
        plt.savefig(f"{report_dir}/proj_eog_{subject}.png", bbox_inches="tight")
        plt.close()

    # Save cleaned data
    raw_ssp.save(output_raw_file, overwrite=True)


# 2 ---------------------------------------

#subjects which need 2 ECG SSPs
subjects = ["sub-0007","sub-0035","sub-0043","sub-0050","sub-0063","sub-0079",
            "sub-0087","sub-0106","sub-0116","sub-0112","sub-0145", "sub-0152",
            "sub-0156","sub-0175","sub-0184","sub-0186","sub-0193","sub-0198",
            "sub-0191","sub-0203","sub-0206","sub-0208","sub-0216","sub-0221",
            "sub-0225","sub-0230","sub-0233","sub-0235","sub-0241","sub-0242",
            "sub-0244","sub-0245","sub-0246", "sub-0162", "sub-0181"]

ssp_preproc_files = []
preproc_files = []
for subject in subjects:
    preproc_files.append(f"{preproc_dir}/{subject}/{subject}_preproc_raw.fif")
    ssp_preproc_files.append(f"{ssp_preproc_dir}/{subject}/{subject}_preproc_raw.fif")

for index in range(len(preproc_files)):
    subject = subjects[index]
    preproc_file = preproc_files[index]
    output_raw_file = ssp_preproc_files[index]

    # Make output directory
    os.makedirs(os.path.dirname(output_raw_file), exist_ok=True)

    # Load preprocessed fif and ICA
    dataset = preprocessing.read_dataset(preproc_file, preload=True)
    raw = dataset["raw"]

    # Only keep MEG, ECG, EOG, EMG
    raw = raw.pick_types(meg=True, ecg=True, eog=True, emg=True)

    # Create a Raw object without any channels marked as bad
    raw_no_bad_channels = raw.copy()
    raw_no_bad_channels.load_bad_channels()

    #  Calculate SSP using ECG
    n_proj = 2
    ecg_epochs = mne.preprocessing.create_ecg_epochs(
        raw_no_bad_channels, picks="all"
    ).average(picks="all")
    ecg_projs, events = mne.preprocessing.compute_proj_ecg(
        raw_no_bad_channels,
        n_grad=n_proj,
        n_mag=n_proj,
        n_eeg=0,
        no_proj=True,
        reject=None,
        n_jobs=6,
    )

    # Add ECG SSPs to Raw object
    raw_ssp = raw.copy()
    raw_ssp.add_proj(ecg_projs.copy())

    # Calculate SSP using EOG
    n_proj = 1
    eog_epochs = mne.preprocessing.create_eog_epochs(
        raw_no_bad_channels, picks="all"
    ).average()
    eog_projs, events = mne.preprocessing.compute_proj_eog(
        raw_no_bad_channels,
        n_grad=n_proj,
        n_mag=n_proj,
        n_eeg=0,
        no_proj=True,
        reject=None,
        n_jobs=6,
    )

    # Add EOG SSPs to Raw object
    raw_ssp.add_proj(eog_projs.copy())

    # Apply SSPs
    raw_ssp.apply_proj()

    # Plot power spectrum of cleaned data
    raw_ssp.plot_psd(fmax=45, n_fft=int(raw.info["sfreq"] * 4))
    plt.savefig(f"{report_dir}/psd_{subject}.png", bbox_inches="tight")
    plt.close()

    if len(ecg_projs) > 0:
        fig = mne.viz.plot_projs_joint(ecg_projs, ecg_epochs, show=False)
        plt.savefig(f"{report_dir}/proj_ecg_{subject}.png", bbox_inches="tight")
        plt.close()

    if len(eog_projs) > 0:
        fig = mne.viz.plot_projs_joint(eog_projs, eog_epochs, show=False)
        plt.savefig(f"{report_dir}/proj_eog_{subject}.png", bbox_inches="tight")
        plt.close()

    # Save cleaned data
    raw_ssp.save(output_raw_file, overwrite=True)
