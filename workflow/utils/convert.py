"""Convert dimer datasets to Dataset objects."""

import pathlib
from typing import Optional

import datasets
import descent.targets.dimers
import descent.targets.energy
import descent.targets.thermo
import descent.utils.loss
import descent.utils.reporting
import openff.interchange
import openff.toolkit
import smee.converters
import torch
import tqdm
from descent.targets.dimers import EnergyFn, create_from_des
from openff.toolkit import ForceField


def get_energy_fn(energy_columns: list[str]) -> EnergyFn:
    """Create an energy function with the specified energy columns."""

    def energy_fn(group_data, ids, coords):
        return torch.tensor(group_data[energy_columns].sum(axis=1).values)

    return energy_fn


def convert_dataset(
    dataset_file_path: str, energy_columns: list[str], output_path: str
) -> None:
    """
    Convert a dataset file to a Dataset object and save it to disk.

    Parameters
    ----------
    dataset_file_path : str
        Path to the dataset file.
    energy_columns : list[str]
        List of energy columns to be used in the energy function.
    output_path : str
        Path where the converted dataset will be saved.
    """
    dataset_file = pathlib.Path(dataset_file_path)
    energy_fn = get_energy_fn(energy_columns)
    dataset = create_from_des(dataset_file, energy_fn=energy_fn)
    dataset.save_to_disk(output_path)


def get_pt_ff_and_tops(
    force_field_path: str,
    output_path: str,
    thermo_dataset_path: Optional[str],
    dimer_dataset_path: Optional[str],
    energy_dataset_path: Optional[str] = None,
) -> None:
    """Get the pytorch force field and topologies from the dataset and force field."""
    smiles_list = []
    for dataset_path, extract_fn in zip(
        [energy_dataset_path, thermo_dataset_path, dimer_dataset_path],
        [
            descent.targets.energy.extract_smiles,
            descent.targets.thermo.extract_smiles,
            descent.targets.dimers.extract_smiles,
        ],
    ):
        if dataset_path is not None:
            dataset = datasets.Dataset.load_from_disk(dataset_path)
            smiles_list.extend(extract_fn(dataset))

    assert smiles_list != [], (
        f"No unique SMILES found in datasets: "
        f"{energy_dataset_path}, {thermo_dataset_path}, {dimer_dataset_path}"
    )

    unique_smiles = set(smiles_list)
    ff = openff.toolkit.ForceField(str(force_field_path), load_plugins=True)

    interchanges = [
        openff.interchange.Interchange.from_smirnoff(
            openff.toolkit.ForceField(str(force_field_path), load_plugins=True),
            openff.toolkit.Molecule.from_mapped_smiles(smiles).to_topology(),
        )
        for smiles in tqdm.tqdm(
            unique_smiles,
            desc="Creating interchanges",
            ncols=80,
            total=len(unique_smiles),
        )
    ]

    force_field, topologies = smee.converters.convert_interchange(interchanges)
    force_field = force_field.to("cuda")

    topologies = {
        smiles: topology.to("cuda")
        for smiles, topology in zip(unique_smiles, topologies)
    }

    for top in topologies.values():  # for some reason needed for hessian calc...
        for param in top.parameters.values():
            param.assignment_matrix = param.assignment_matrix.to_dense()

    torch.save((force_field, topologies), pathlib.Path(output_path))


def convert_to_offxml(
    base_ff: pathlib.Path, smee_forcefield: smee.TensorForceField, output: pathlib.Path
):
    """Convert the final force field parameters into an offxml file."""
    # need load plugins for dexp
    ff: ForceField = ForceField(base_ff, load_plugins=True)
    descent.utils.reporting.print_force_field_summary(smee_forcefield)

    for potential in smee_forcefield.potentials:
        ff_handler_name = potential.parameter_keys[0].associated_handler
        if potential.type != "vdW":
            continue
        ff_handler = ff.get_parameter_handler(ff_handler_name)

        # check if we have handler attributes to update
        attribute_names = potential.attribute_cols
        attribute_units = potential.attribute_units

        if potential.attributes is not None:
            opt_attributes = potential.attributes.detach().cpu().numpy()
            for j, (p, unit) in enumerate(zip(attribute_names, attribute_units)):
                setattr(ff_handler, p, opt_attributes[j] * unit)

        parameter_names = potential.parameter_cols
        parameter_units = potential.parameter_units

        for i in range(len(potential.parameters)):
            smirks = potential.parameter_keys[i].id
            if "EP" in smirks:
                print(f"Skipping {smirks} as it is a virtual site")
                # skip fitted sites to dimers, we only have water and it should be 0 anyway
                continue
            ff_parameter = ff_handler[smirks]
            opt_parameters = potential.parameters[i].detach().cpu().numpy()
            for j, (p, unit) in enumerate(zip(parameter_names, parameter_units)):
                setattr(ff_parameter, p, opt_parameters[j] * unit)

    ff.to_file(output)


def to_vdw_only_ff(ff: smee.TensorForceField) -> smee.TensorForceField:
    return smee.TensorForceField(
        potentials=[ff.potentials_by_type["vdW"]], v_sites=ff.v_sites
    )
