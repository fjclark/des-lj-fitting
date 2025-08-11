"""Functionality for plotting training/ testing results."""

from pathlib import Path

import bokeh.io
import bokeh.palettes
import bokeh.plotting
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotmol
from openff.toolkit import ForceField
from openff.toolkit.typing.engines.smirnoff.parameters import vdWType
from scipy.stats import linregress
from tbparse import SummaryReader

from .models import WorkflowConfig

plt.style.use("ggplot")


def plot_loss(configs: list[WorkflowConfig], output_path: Path) -> None:
    """Plot the training and test total, force, and energy loss."""

    dfs = {
        config.experiment_name: SummaryReader(config.fit_dir).scalars
        for config in configs
    }

    # Three plots on one level
    with plt.style.context("ggplot"):
        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        scalar_names = {"Total Loss": "Loss"}

        for i, (title, scalar) in enumerate(scalar_names.items()):
            for experiment_name, df in dfs.items():
                df_filtered = df[df["tag"] == scalar]
                ax.plot(
                    df_filtered["step"],
                    df_filtered["value"],
                    label=experiment_name,
                    alpha=0.8,
                )

            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            if i == 2:
                ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.0))

        fig.tight_layout()
        fig.savefig(str(output_path), dpi=300)
        plt.close(fig)


def get_changed_vdw_parameters(
    ff_original: ForceField, ff_fitted: ForceField
) -> list[tuple[vdWType, vdWType]]:
    """
    Compare the original and fitted force fields and return a list of changed parameters.
    """
    changed_parameters = []

    for parameter_orig, parameter_fit in zip(
        ff_original._parameter_handlers["vdW"].parameters,
        ff_fitted._parameter_handlers["vdW"].parameters,
        strict=True,
    ):
        if parameter_orig.to_dict() != parameter_fit.to_dict():
            changed_parameters.append((parameter_orig, parameter_fit))

    return changed_parameters


def plot_epsilons(
    changed_parameters: list[tuple[vdWType, vdWType]], ax: plt.Axes
) -> None:
    """Plot the initial and final epsilons of the changed parameters."""
    initial_epsilon = [
        parameter_orig.epsilon.m_as("kilocalorie_per_mole")
        for parameter_orig, _ in changed_parameters
    ]
    final_epsilon = [
        parameter_fit.epsilon.m_as("kilocalorie_per_mole")
        for _, parameter_fit in changed_parameters
    ]
    smirks = [parameter_orig.smirks for parameter_orig, _ in changed_parameters]

    # Bar plot
    offset = 0.2
    # Plot the initial and final epsilons for the same smirks
    ax.bar(
        range(len(initial_epsilon)),
        initial_epsilon,
        width=offset,
        label="Initial epsilon",
        color="blue",
        alpha=0.5,
    )
    ax.bar(
        [i + offset for i in range(len(final_epsilon))],
        final_epsilon,
        width=offset,
        label="Final epsilon",
        color="orange",
        alpha=0.5,
    )
    # Set the x-ticks to be the smirks
    ax.set_xticks([i + offset / 2 for i in range(len(smirks))])
    ax.set_xticklabels(smirks, rotation=90)
    # Set the y-axis label
    ax.set_ylabel(r"Epsilon / $\mathrm{kcal \cdot mol^{-1}}$")
    # Set the title
    ax.set_title("Epsilon changes in the fitted force field")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")


def plot_sigma(changed_parameters: list[tuple[vdWType, vdWType]], ax: plt.Axes) -> None:
    """Plot the initial and final sigmas of the changed parameters."""
    initial_sigma = [
        parameter_orig.sigma.m_as("angstrom")
        for parameter_orig, _ in changed_parameters
    ]
    final_sigma = [
        parameter_fit.sigma.m_as("angstrom") for _, parameter_fit in changed_parameters
    ]
    smirks = [parameter_orig.smirks for parameter_orig, _ in changed_parameters]
    # Bar plot
    offset = 0.2
    # Plot the initial and final sigmas for the same smirks
    ax.bar(
        range(len(initial_sigma)),
        initial_sigma,
        width=offset,
        label="Initial sigma",
        color="blue",
        alpha=0.5,
    )
    ax.bar(
        [i + offset for i in range(len(final_sigma))],
        final_sigma,
        width=offset,
        label="Final sigma",
        color="orange",
        alpha=0.5,
    )
    # Set the x-ticks to be the smirks
    ax.set_xticks([i + offset / 2 for i in range(len(smirks))])
    ax.set_xticklabels(smirks, rotation=90)
    # Set the y-axis label
    ax.set_ylabel("Sigma / $\mathrm{\\AA}$")
    # Set the title
    ax.set_title("Sigma changes in the fitted force field")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")


def plot_vdw_parameter_changes_from_parameters(
    changed_parameters: list[tuple[vdWType, vdWType]],
) -> tuple[plt.Figure, plt.Axes]:
    """Plot the changes in epsilon and sigma for the changed parameters."""
    # Scale the figure size by the number of changed parameters
    num_parameters = len(changed_parameters)
    fig, ax = plt.subplots(2, 1, figsize=(num_parameters * 0.6, 12))
    plot_epsilons(changed_parameters, ax[0])
    # Remove x labels from the first plot
    ax[0].set_xlabel("")
    # Also remove tick labels
    ax[0].set_xticklabels([])
    plot_sigma(changed_parameters, ax[1])
    fig.tight_layout()

    return fig, ax


def plot_vdw_parameter_changes(
    original_ff: ForceField, fitted_ff: ForceField, output_path: Path
) -> None:
    """
    Plot the changes in epsilon and sigma for the changed parameters in the original and fitted force fields.
    """
    changed_parameters = get_changed_vdw_parameters(original_ff, fitted_ff)
    fig, ax = plot_vdw_parameter_changes_from_parameters(changed_parameters)
    fig.savefig(str(output_path), dpi=300)
    plt.close(fig)


def create_interactive_plot(data: pd.DataFrame, output_path: Path) -> None:
    """
    Interactive plot with points split vertically in two colours (left/right halves).
    Colours correspond to smiles_a and smiles_b.
    """

    # Check type consistency
    assert all(
        data["type"] == data["type"].iloc[0]
    ), "All entries in the 'type' column must be the same."
    entry_type = data["type"].iloc[0]

    figure = bokeh.plotting.figure(
        tooltips=plotmol.default_tooltip_template(),
        x_axis_label=f"Reference {entry_type}",
        y_axis_label=f"Predicted {entry_type}",
        width=600,
        height=600,
    )

    # Generate colours for all unique smiles
    unique_smiles = pd.unique(data[["smiles_a", "smiles_b"]].values.ravel())
    palette = bokeh.palettes.viridis(len(unique_smiles))
    smiles_to_color = {sm: palette[i] for i, sm in enumerate(unique_smiles)}

    df = data.copy()
    df["color_a"] = [smiles_to_color[sm] for sm in df["smiles_a"]]
    df["color_b"] = [smiles_to_color[sm] for sm in df["smiles_b"]]
    df["combined_smiles"] = [
        ".".join([a, b]) for a, b in zip(df["smiles_a"], df["smiles_b"])
    ]

    # Circle radius (in data units) — adjust as needed
    radius = (df["ref"].max() - df["ref"].min()) * 0.01

    # Plot molecule tooltips for full circles (invisible)
    # Use plotmol.scatter just to get molecule hover, but alpha=0
    plotmol.scatter(
        figure,
        x=df["ref"],
        y=df["pred"],
        smiles=df["combined_smiles"],
        marker_size=30,
        marker_color="white",
        alpha=0,
    )

    # Add left half wedges for smiles_a colours
    figure.wedge(
        x=df["ref"],
        y=df["pred"],
        radius=radius,
        start_angle=np.pi / 2,  # 90 degrees
        end_angle=3 * np.pi / 2,  # 270 degrees
        fill_color=df["color_a"],
        line_color=None,
        alpha=0.9,
    )

    # Add right half wedges for smiles_b colours
    figure.wedge(
        x=df["ref"],
        y=df["pred"],
        radius=radius,
        start_angle=3 * np.pi / 2,  # 270 degrees
        end_angle=np.pi / 2,  # 90 degrees (wraps around)
        fill_color=df["color_b"],
        line_color=None,
        alpha=0.9,
    )

    # Leave uncertainty for know until we implement standard errors
    # figure.segment(
    #     x0=df["ref"],
    #     y0=df["pred"] - df["std_pred"],
    #     x1=df["ref"],
    #     y1=df["pred"] + df["std_pred"],
    #     line_color="black",
    #     line_width=1,
    #     alpha=0.7,
    #     legend_label="Prediction Uncertainty",
    # )

    # Perfect prediction line
    min_val = min(df["ref"].min(), df["pred"].min())
    max_val = max(df["ref"].max(), df["pred"].max())
    figure.line(
        [min_val, max_val],
        [min_val, max_val],
        line_color="black",
        line_width=2,
        alpha=0.5,
        legend_label="x = y",
    )

    # Line of best fit
    slope, intercept, r_value, _, _ = linregress(df["ref"], df["pred"])
    x_fit = df["ref"]
    y_fit = slope * x_fit + intercept
    figure.line(x_fit, y_fit, legend_label=f"Best Fit (R² = {r_value**2:.2f})")

    # Configure legend
    figure.legend.location = "top_left"
    figure.legend.click_policy = "hide"

    # Show + save
    bokeh.plotting.show(figure)
    bokeh.io.save(figure, filename=output_path)
