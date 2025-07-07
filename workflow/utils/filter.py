"""
Filter the DES dataset to only contain records with elements of interest
"""

import pandas as pd
from loguru import logger
from rdkit import Chem


def unwanted_molecules_filter(
    row: pd.Series, elements_to_keep: list[str], ions_to_remove: list[str]
) -> bool:
    """
    Pandas filter function which returns True if the record contains unwanted elements,
    ions or charged molecules.
    """
    for element in row["elements"].split():
        if element not in elements_to_keep:
            return True

    # check for ions
    for smiles in [row["smiles0"], row["smiles1"]]:
        if smiles in ions_to_remove:
            return True

    # slow check last - remove charged
    for smiles in [row["smiles0"], row["smiles1"]]:
        rdkit_mol = Chem.MolFromSmiles(smiles)
        total_charge = sum([atom.GetFormalCharge() for atom in rdkit_mol.GetAtoms()])
        if total_charge != 0:
            return True

    return False


def filter_dataset(
    dataset_csv_path: str,
    elements_to_keep: list[str],
    ions_to_remove: list[str],
    output_csv_path: str,
) -> None:
    """
    Filter the dataset to only contain records with elements of interest and no unwanted ions or charged molecules.

    Parameters
    ----------
    dataset_csv_path : str
        Path to the input dataset CSV file.
    elements_to_keep : list[str]
        List of elements to keep in the dataset.
    ions_to_remove : list[str]
        List of ions to remove from the dataset.
    output_csv_path : str
        Path to save the filtered dataset CSV file.
    """
    raw_dataset = pd.read_csv(dataset_csv_path)
    logger.info(
        f"Loaded dataset with {len(raw_dataset)} records and {len(raw_dataset['system_id'].unique())} unique systems."
    )

    unwanted_rows = raw_dataset.apply(
        unwanted_molecules_filter, axis=1, args=(elements_to_keep, ions_to_remove)
    )

    filtered_dataset = raw_dataset[~unwanted_rows]
    logger.info(f"Filtered dataset to {len(filtered_dataset)} records.")

    filtered_dataset.to_csv(output_csv_path, index=False)

    # work out the total number of unique systems and the total number of data points
    logger.info(
        f"Total number of records after filtering: {len(filtered_dataset)} and unique systems: {len(filtered_dataset['system_id'].unique())}"
    )
