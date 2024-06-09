"""Perform dipole sign flipping.

"""

from glob import glob
from dask.distributed import Client

from osl import utils
from osl.source_recon import find_template_subject, run_src_batch


if __name__ == "__main__":
    utils.logger.set_up(level="INFO")
    client = Client(n_workers=6, threads_per_worker=1)

    src_dir = "/home/mtrubshaw/Documents/ALS_dyn/data/src"

    subjects = []
    for path in sorted(glob(f"{src_dir}/*/parc/parc-raw.fif")):
        subject = path.split("/")[-3]
        subjects.append(subject)

    template = find_template_subject(
        src_dir, subjects, n_embeddings=15, standardize=True
    )

    config = f"""
        source_recon:
        - fix_sign_ambiguity:
            template: {template}
            n_embeddings: 15
            standardize: True
            n_init: 3
            n_iter: 2500
            max_flips: 20
    """

    run_src_batch(
        config,
        src_dir=src_dir,
        subjects=subjects,
        dask_client=True,
    )
