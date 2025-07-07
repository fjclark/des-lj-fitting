"""Functions to download and prepare data for the DES-LJ fitting workflow."""

import subprocess


def download_DES370K_dimer_data() -> None:
    """Download the DES370K dimer dataset from Zenodo."""
    cmds = [
        ["mkdir", "-p", "tmp"],
        [
            "curl",
            "https://zenodo.org/records/5676266/files/DES370K.zip?download=1",
            "-o",
            "tmp/DES370K.zip",
        ],
        ["unzip", "tmp/DES370K.zip", "-d", "tmp/DES370K"],
        ["mv", "tmp/DES370K/geometries", "dimer_datasets/DES370K"],
        ["rm", "-rf", "tmp"],
    ]
    # Run with shell = False
    for cmd in cmds:
        subprocess.run(cmd, shell=False, check=True)
