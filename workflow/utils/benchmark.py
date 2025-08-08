"""Functionality for benchmarking the fitted force fields."""

from pathlib import Path

import datasets
import descent.train
import pandas as pd
import torch
from loguru import logger

from .convert import get_pt_ff_and_tops
from .models import WorkflowConfig
from .simulate import iter_predicted_properties, run_required_simulations


def benchmark_thermo(
    config: WorkflowConfig,
    thermo_dataset_path: str,
    output_dir_path: str,
) -> None:
    """Benchmark the fitted force field against condensed phase properties."""

    logger.info(f"Benchmarking {config.experiment_name} on {thermo_dataset_path}")

    # Create the required directories
    output_dir_pathlib_path = Path(output_dir_path)
    output_dir_pathlib_path.mkdir(exist_ok=True, parents=True)

    ffs_and_tops_path = output_dir_pathlib_path / "ffs_and_tops.pt"
    output_table_path = output_dir_pathlib_path / "output_table.csv"

    simulation_output_path = output_dir_pathlib_path / "simulation_output"
    simulation_output_path.mkdir(exist_ok=True)

    # Create the topologies if required
    if not ffs_and_tops_path.exists():
        logger.info(f"Generating force field and topologies at {ffs_and_tops_path}")
        get_pt_ff_and_tops(
            force_field_path=str(config.output_ff_path),
            output_path=str(ffs_and_tops_path),
            thermo_dataset_path=thermo_dataset_path,
            dimer_dataset_path=None,
        )

    logger.info(f"Loaded force field and topologies from {ffs_and_tops_path}")
    tensor_ff, topologies = torch.load(ffs_and_tops_path, map_location="cuda")

    logger.info(f"Loading thermo dataset from {thermo_dataset_path}")
    thermo_dataset = datasets.load_from_disk(thermo_dataset_path)

    trainable = descent.train.Trainable(
        tensor_ff,
        parameters={},
        attributes={},
    )
    x = trainable.to_values()

    # Plan the required simulations
    entries = [*descent.utils.dataset.iter_dataset(thermo_dataset)]
    required_simulations, entry_to_simulation = (
        descent.targets.thermo._plan_simulations(entries, topologies)
    )
    logger.info(f"Required simulations: {required_simulations.keys()}")

    # Run required simulations
    logger.info(f"Running {len(required_simulations)} required simulations")
    frames = run_required_simulations(
        trainable,
        x,
        required_simulations,
        simulation_output_path,
        max_workers=2,
    )

    # Calculate the loss from the predicted properties
    results_table = []
    for entry, pred, std in iter_predicted_properties(
        entries,
        entry_to_simulation,
        required_simulations,
        frames,
        trainable,
        x,
    ):
        std_ref = "" if entry["std"] is None else f" ± {float(entry['std']):.3f}"
        results_table.append(
            {
                "type": f"{entry['type']} [{entry['units']}]",
                "smiles_a": descent.utils.molecule.unmap_smiles(entry["smiles_a"]),
                "smiles_b": (
                    ""
                    if entry["smiles_b"] is None
                    else descent.utils.molecule.unmap_smiles(entry["smiles_b"])
                ),
                "pred": f"{float(pred.item()):.3f} ± {float(std.item()):.3f}",
                "ref": f"{float(entry['value']):.3f}{std_ref}",
            }
        )

    df = pd.DataFrame(results_table)
    logger.info(f"Results table:\n{df}")
    logger.info(f"Saving results table to {output_table_path}")
    df.to_csv(output_table_path)
