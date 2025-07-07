"""Functionality for plotting training/ testing results."""

from pathlib import Path

import matplotlib.pyplot as plt
from tbparse import SummaryReader

from .models import WorkflowConfig


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
