"""Preprocess MaxFiltered sensor data.

"""

import pandas as pd
from dask.distributed import Client

from osl import preprocessing, utils


if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=6, threads_per_worker=1)

    config = """
        preproc:
        - filter: {l_freq: 0.5, h_freq: 125, method: iir, iir_params: {order: 5, ftype: butter}}
        - notch_filter: {freqs: 50 100}
        - resample: {sfreq: 250}
        - bad_segments: {segment_len: 500, picks: mag, significance_level: 0.1}
        - bad_segments: {segment_len: 500, picks: grad, significance_level: 0.1}
        - bad_segments: {segment_len: 500, picks: mag, mode: diff, significance_level: 0.1}
        - bad_segments: {segment_len: 500, picks: grad, mode: diff, significance_level: 0.1}
        - bad_channels: {picks: mag, significance_level: 0.1}
        - bad_channels: {picks: grad, significance_level: 0.1}
        - ica_raw: {picks: meg, n_components: 40}
        - ica_autoreject: {apply: False}
        - interpolate_bads: {}
    """

    maxfilter_dir = "../data/maxfiltered"
    preproc_dir = "../data/preproc"

    participants = pd.read_csv("../data/demographics/complete_demo_dyn_als.csv.csv")

    subjects = participants["Subject"].values
    raw_files = participants["Raw_File"].values

    maxfiltered_files = []
    for raw_file in raw_files:
        mf_file = raw_file.replace(".fif", "_tsss.fif")
        maxfiltered_files.append(f"{maxfilter_dir}/"+mf_file)

    preprocessing.run_proc_batch(
        config,
        maxfiltered_files,
        outdir=preproc_dir,
        outnames=subjects,
        overwrite=True,
        dask_client=True,
    )
