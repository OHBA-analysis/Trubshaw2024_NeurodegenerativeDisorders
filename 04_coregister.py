"""Coregistration.

"""

import pandas as pd
from glob import glob
from dask.distributed import Client

from osl import source_recon, utils


# if using standard brain (missing structural) remember to add allow_smri_scaling: True to config under coregister

if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    source_recon.setup_fsl(fsl_dir)
    client = Client(n_workers=6, threads_per_worker=1)

    config = """
        source_recon:
        - extract_fiducials_from_fif: {}
        - remove_stray_headshape_points: {}
        - compute_surfaces:
            include_nose: False
        - coregister:
            use_nose: False
            use_headshape: True
        - forward_model:
            model: Single Layer
    """

    preproc_dir = "/home/mtrubshaw/Documents/ALS_dyn/data/preproc_ssp"
    smri_dir = "/home/mtrubshaw/Documents/ALS_dyn/data/smri"
    coreg_dir = "/home/mtrubshaw/Documents/ALS_dyn/data/coreg"
    

    participants = pd.read_csv("/home/mtrubshaw/Documents/ALS_dyn/data/demographics/complete_demo_dyn_als.csv")
    subjects = participants["Subject"].values
    datasets = participants["Dataset"].values
    structurals = participants["Structural"].values

    preproc_files = []
    smri_files = []
    for subject, structural in zip(subjects, structurals):
        preproc_files.append(f"{preproc_dir}/{subject}/{subject}_preproc_raw.fif")
        smri_files.append(f"{smri_dir}/{structural}")

    source_recon.run_src_batch(
        config,
        src_dir=coreg_dir,
        subjects=subjects,
        preproc_files=preproc_files,
        smri_files=smri_files,
        dask_client=True,
    )
