import pickle as pkl
from copy import deepcopy
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from ase import Atoms
from datasets import Dataset
from loguru import logger
from mace.calculators import mace_off, mace_omol
from openff.toolkit import Molecule
from rdkit import Chem
from tqdm import tqdm

plt.style.use("ggplot")

EV_TO_KCALMOL = 23.0605
DIMER_DATASET_PATH = Path("../dimer_datasets/DES370K_cbs_CCSD(T)_all_dataset")

MACE_MODELS = ["small", "medium", "large", "extra_large"]
MODELS_TO_CALCS = {
    "small": mace_off,
    "medium": mace_off,
    "large": mace_off,
    "extra_large": mace_omol,
}


def is_dimer_scan(row: pd.Series, orig_data: pd.DataFrame) -> bool:
    group_id = int(row["source"].split(" ")[-1].split("=")[1])
    # Get the row from the original data
    orig_data_row = orig_data[orig_data["group_id"] == group_id]
    # Now extract the group_orig
    group_origs = orig_data_row["group_orig"].values
    return all(["dimer" in group_orig for group_orig in group_origs])


def to_ase(mol_a: Chem.Mol, mol_b: Chem.Mol, coords: npt.NDArray) -> Atoms:
    """Convert mapped SMILES + coordinates into an ASE Atoms object."""

    symbols, tags = [], []
    for mol in mol_a, mol_b:
        for atom in mol.GetAtoms():
            symbols.append(atom.GetSymbol())
            tags.append(atom.GetAtomMapNum())  # mapping numbers

    # Build ASE Atoms
    atoms = Atoms(symbols=symbols, positions=coords)
    atoms.set_tags(tags)

    # All neutral, singlet in this dataset
    atoms.info["charge"] = 0.0  # neutral
    atoms.info["spin"] = 1.0  # spin multiplicity

    return atoms


def get_energy_mace(
    smiles_a: str, smiles_b: str, coords: torch.Tensor, model_name: str
) -> npt.NDArray:
    """Evaluate the energies for a set of dimer coordinates using MACE-OFF"""

    # Convert
    mol_a = Molecule.from_mapped_smiles(smiles_a).to_rdkit()
    mol_b = Molecule.from_mapped_smiles(smiles_b).to_rdkit()

    # Reshape the coordinates
    n_atoms_a, n_atoms_b = mol_a.GetNumAtoms(), mol_b.GetNumAtoms()
    tot_atoms = n_atoms_a + n_atoms_b
    coords = coords.view(-1, tot_atoms, 3)
    coords = coords.detach().cpu().numpy()

    # Add a single new set of coordinates at massive separation to allow the definition
    # of 0 energy. Do this by adding 10,000 A to the second molecule z direction
    coords_extended = deepcopy(coords[-1])
    coords_extended[n_atoms_a:, 2] += 10000
    coords = np.concatenate([coords, coords_extended[np.newaxis, ...]], axis=0)

    # Get the MACE calculator
    calc = MODELS_TO_CALCS[model_name](model=model_name, device="cuda")

    # Get the energies
    energies = np.zeros(coords.shape[0])
    for i, coord in enumerate(coords):
        atoms = to_ase(mol_a, mol_b, coord)
        atoms.calc = calc
        energies[i] = atoms.get_potential_energy() * EV_TO_KCALMOL

    return energies


def main():
    # Load datasets
    dataset_all = Dataset.load_from_disk(DIMER_DATASET_PATH)
    orig_data = pd.read_csv("DES370K.csv")
    filter_fn = partial(is_dimer_scan, orig_data=orig_data)
    dataset_dimer = dataset_all.filter(filter_fn)
    logger.info(f"{len(dataset_all)=}")
    logger.info(f"{len(dataset_dimer)=}")

    for model in MACE_MODELS:
        logger.info(f"Evaluating MACE model: {model}")
        energies_mace_all = []
        for entry in tqdm(dataset_dimer):
            energies_mace = get_energy_mace(
                entry["smiles_a"], entry["smiles_b"], entry["coords"], model_name=model
            )
            energies_mace_all.append(energies_mace)

        # Save the energies
        with open(f"energies_mace_{model}.pkl", "wb") as f:
            pkl.dump(energies_mace_all, f)


if __name__ == "__main__":
    main()
