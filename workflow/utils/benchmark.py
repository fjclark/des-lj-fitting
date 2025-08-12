"""Functionality for benchmarking the fitted force fields."""

from pathlib import Path

import datasets
import descent.train
import numpy as np
import pandas as pd
import panel as pn
import torch
from loguru import logger
from tqdm import tqdm

from . import stats
from .convert import get_pt_ff_and_tops
from .models import WorkflowConfig
from .plot import create_interactive_plot
from .simulate import iter_predicted_properties, run_required_simulations


def get_results_table(
    entries: list,
    entry_to_simulation: list,
    required_simulations: dict,
    frames: dict,
    trainable: descent.train.Trainable,
    x: torch.Tensor,
) -> pd.DataFrame:
    """Generate a DataFrame of predicted and reference properties."""

    results_table = []
    for entry, pred, std in tqdm(
        iter_predicted_properties(
            entries,
            entry_to_simulation,
            required_simulations,
            frames,
            trainable,
            x,
        ),
        desc="Calculating thermo properties",
        total=len(entries),
    ):
        results_table.append(
            {
                "type": f"{entry['type']} [{entry['units']}]",
                "smiles_a": descent.utils.molecule.unmap_smiles(entry["smiles_a"]),
                "smiles_b": (
                    ""
                    if entry["smiles_b"] is None
                    else descent.utils.molecule.unmap_smiles(entry["smiles_b"])
                ),
                "pred": float(pred.item()),
                "std_pred": float(std.item()),
                "ref": float(entry["value"]),
                "std_ref": np.nan if entry["std"] is None else float(entry["std"]),
            }
        )
    return pd.DataFrame(results_table)


def get_metrics(
    df: pd.DataFrame,
    output_dir_pathlib_path: Path,
) -> pd.DataFrame:
    """Compute summary statistics for each entry type and save to CSV."""

    metrics: dict[str, dict[str, float]] = {}
    for entry_type in df["type"].unique():
        metrics[entry_type] = {}
        df_type = df[df["type"] == entry_type]
        logger.info(f"Summary statistics for {entry_type}:")
        for metric_name, metric_fn in stats.METRIC_FNS.items():
            y_ref = df_type["ref"].to_numpy(dtype=float)
            y_pred = df_type["pred"].to_numpy(dtype=float)
            value = metric_fn(y_pred, y_ref)
            ci_95 = stats.get_bootstrapped_metric(metric_fn, y_pred, y_ref)
            logger.info(
                f"{metric_name}: {value:.3f} [{ci_95[0]:.3f}, {ci_95[1]:.3f}, {ci_95[2]:.3f}]"
            )
            metrics[entry_type][metric_name] = value
            metrics[entry_type][f"{metric_name}_bootstrap_mean"] = ci_95[0]
            metrics[entry_type][f"{metric_name}_ci_95_lower"] = ci_95[1]
            metrics[entry_type][f"{metric_name}_ci_95_upper"] = ci_95[2]

    metrics_df = pd.DataFrame(metrics).T
    metrics_df.index.name = "metric"
    metrics_df.columns.name = "type"
    metrics_output_path = output_dir_pathlib_path / "metrics.csv"
    logger.info(f"Saving metrics to {metrics_output_path}")
    logger.info(f"Metrics DataFrame:\n{metrics_df}")
    metrics_df.to_csv(metrics_output_path)
    return metrics_df


def format_metrics_df_for_human(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """Format the metrics DataFrame for human readability."""
    formatted_df_data: dict[str, dict[str, str]] = {}
    for entry_type, metrics in metrics_df.iterrows():
        formatted_df_data[entry_type] = {}
        for metric in stats.METRIC_FNS.keys():
            formatted_df_data[entry_type][
                metric
            ] = f"{metrics[metric]:.3f} [{metrics[f'{metric}_ci_95_lower']:.3f} {metrics[f'{metric}_ci_95_upper']:.3f}]"

    formatted_df = pd.DataFrame(formatted_df_data).T
    formatted_df.index.name = "metric"
    formatted_df.columns.name = "type"
    return formatted_df


def make_summary_table(metrics_df: pd.DataFrame, output_path: Path) -> None:
    """Create and save a Panel summary table from the metrics DataFrame."""

    pn.extension()

    metrics_panel = pn.panel(metrics_df)
    metrics_panel.save(output_path)


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
    output_table_path = output_dir_pathlib_path / "results_table.csv"

    simulation_output_path = output_dir_pathlib_path / "simulation_output"
    simulation_output_path.mkdir(exist_ok=True)

    # Create the topologies if required
    if not ffs_and_tops_path.exists():
        logger.info(
            f"Generating force field and topologies at {ffs_and_tops_path} from {config.output_ff_path}"
        )
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
    logger.info(f"Required simulations bulk: {required_simulations['bulk'].keys()}")
    logger.info(f"Required simulations vacuum: {required_simulations['vacuum'].keys()}")

    # Run required simulations
    logger.info(
        f"Running {len(required_simulations['bulk'])} required bulk simulations and {len(required_simulations['vacuum'])} required vacuum simulations"
    )
    frames = run_required_simulations(
        trainable,
        x,
        required_simulations,
        simulation_output_path,
        benchmarking_config=config.benchmarking,
        max_workers=2,
    )

    df = get_results_table(
        entries,
        entry_to_simulation,
        required_simulations,
        frames,
        trainable,
        x,
    )

    logger.info(f"Results table:\n{df}")
    logger.info(f"Saving results table to {output_table_path}")
    df.to_csv(output_table_path)

    metrics_df = get_metrics(df, output_dir_pathlib_path)

    make_summary_table(
        format_metrics_df_for_human(metrics_df),
        output_dir_pathlib_path / "summary_metrics.html",
    )

    # Create interactive plots for each data type
    for entry_type in df["type"].unique():
        entry_df = df[df["type"] == entry_type]
        entry_df = entry_df.sort_values(by=["smiles_a", "smiles_b"])
        output_path = (
            output_dir_pathlib_path / f"interactive_plot_{entry_type.split()[0]}.html"
        )
        logger.info(f"Creating interactive plot for {entry_type} at {output_path}")
        create_interactive_plot(entry_df, output_path)
