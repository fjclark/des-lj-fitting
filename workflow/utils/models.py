"""Pydantic models. These will be stored as, and read from yaml files."""

from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, Field


class DimerConfig(BaseModel):
    raw_data_dir: Path = Field(
        default="dimer_datasets/DES370K",
        description="Directory where the raw dimer data is stored.",
    )
    get_data_fn: str = Field(
        default="utils.download_data.download_DES370K_dimer_data",
        description="Function to get the dimer data.",
    )
    energy_columns: list[str] = Field(
        default=["ANA2B_V_ana", "ANA2B_V_D3"],
        description="Energy columns to sum to get the target energy for the dimer data.",
    )

    loss_fn: str = Field(
        default="descent.utils.loss.get_loss_dimer_boltz_ref_0",
        description="Loss function to use for training.",
    )

    vdw_only: bool = Field(
        default=True,
        description="Whether we are fitting only to vdW interactions. Expected to be the case if using the ANA2B decomposition.",
    )

    elements_to_keep: list[str] = Field(
        default=["H", "C", "N", "O", "F", "S", "Cl"],
        description="Elements to keep in the dimer dataset.",
    )

    ions_to_remove: list[str] = Field(
        default=["[F-]", "[Cl-]", "[H][H]"],
        description="Ions to remove from the dimer dataset. Most will be caught by the element filter - just need to remove some halogen ones.",
    )

    @property
    def raw_geometries_dir(self) -> Path:
        return self.raw_data_dir / "geometries"

    @property
    def processed_data_dir(self) -> Path:
        return Path(
            str(self.raw_data_dir) + f"_{'_'.join(self.energy_columns)}_" + "dataset"
        )

    @property
    def dataset_name(self) -> str:
        return self.raw_data_dir.name

    @property
    def processed_csv_path(self) -> Path:
        return self.raw_data_dir / f"{self.dataset_name}.csv"

    @property
    def raw_csv_path(self) -> Path:
        return self.raw_data_dir / f"{self.dataset_name}_all.csv"


class ThermoConfig(BaseModel):
    """Configuration for the thermodynamic data fitting."""

    # Add fields as needed for condensed phase
    raw_data_dir: Path = Field(
        default="thermo_datasets/raw",
        description="Directory where the raw condensed phase data is stored.",
    )
    # Add more fields as required
    # Add processed_data_dir...


class TrainingConfig(BaseModel):
    """Configuration for the training process."""

    starting_force_field: str = Field(
        default="openff_unconstrained-2.2.0",
        description="Name of the starting force field to use for fitting.",
    )

    weights: dict[str, float] = Field(
        default_factory=lambda: {
            "dimer": 1.0,
            "thermo": 1.0,
        },
        description="Weights for the different datasets in the loss function.",
    )

    thermo_scales: dict[str, float] = Field(
        default_factory=lambda: {
            "density": 50,
            "hmix": 10,
        },
        description="Scales for the thermodynamic properties in the loss function.",
    )

    parameters: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {
            "vdW": {
                "cols": ["epsilon", "sigma"],
                "scales": {"epsilon": 10, "sigma": 1.0},
                "limits": {"epsilon": [0.0, None], "sigma": [0.0, None]},
                "exclude": ["[#1:1]-[#8X2H2+0]-[#1]"],  # Exclude water
            },
        },
        description="Trainable parameters for the force field.",
    )

    attributes: dict[str, dict[str, Any]] = Field(
        default_factory=lambda: {},
        description="Trainable attributes for the force field.",
    )

    @property
    def starting_force_field_path(self) -> Path:
        return Path(f"input_ff/{self.starting_force_field}.offxml")


class WorkflowConfig(BaseModel):
    experiment_name: str = Field(default="", description="Name of the experiment.")
    experiment_description: str = Field(
        default="", description="Description of the experiment."
    )

    dimer: Optional[DimerConfig] = Field(
        default=None, description="Dimer fitting configuration."
    )
    thermo: Optional[ThermoConfig] = Field(
        default=None, description="Thermodynamic property fitting configuration."
    )
    training: TrainingConfig = Field(
        default_factory=TrainingConfig,
        description="Configuration for the training process.",
    )

    @property
    def pt_ff_and_tops_path(self) -> Path:
        """Combine the dimer data, condensed phase data, and the starting force field
        to create a path for the PyTorch force field and topologies file."""

        name = self.training.starting_force_field
        if self.dimer:
            name += f"_{self.dimer.processed_data_dir.name}"
        if self.thermo:
            name += f"_{self.thermo.processed_data_dir.name}"

        return Path(f"ff_and_top_files/{name}_ff_and_tops.pt")

    @property
    def fit_dir(self) -> Path:
        """Directory where the training will take place."""
        return Path(f"fits/{self.experiment_name}")

    @property
    def output_description_path(self) -> Path:
        """Path to the output description file."""
        return self.fit_dir / f"{self.experiment_name}.txt"

    @property
    def output_ff_path(self) -> Path:
        """Path to the output force field file."""
        return Path("output_ff", f"{self.experiment_name}.offxml")

    @classmethod
    def from_file(cls, filename: str | Path) -> "WorkflowConfig":
        with open(filename, "r") as f:
            data = yaml.safe_load(f)
            return cls(**data)

    def to_file(self, filename: str | Path):
        with open(filename, "w") as f:
            yaml.dump(self.dict(), f, default_flow_style=False, sort_keys=False)
