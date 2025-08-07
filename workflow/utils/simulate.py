"""Functionality for running condensed phase simulations."""

import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterator, Tuple

import descent.optim
import descent.targets
import descent.targets.dimers
import descent.targets.thermo
import descent.train
import descent.utils.loss
import descent.utils.molecule
import descent.utils.reporting
import openmm.unit
import smee
import torch
import tqdm
from loguru import logger


# From https://github.com/jthorton/descent/blob/8aeaaee88d6192525891ea7ffbf7a8ff0e8c09e1/descent/targets/thermo.py#L307
def _bulk_config(
    temperature: float, pressure: float
) -> descent.targets.thermo.SimulationConfig:
    """Return a default simulation configuration for simulations of the bulk phase.

    Args:
        temperature: The temperature [K] at which to run the simulation.
        pressure: The pressure [atm] at which to run the simulation.

    Returns:
        The default simulation configuration.
    """
    temperature = temperature * openmm.unit.kelvin
    pressure = pressure * openmm.unit.atmosphere

    return descent.targets.thermo.SimulationConfig(
        max_mols=1000,
        gen_coords=smee.mm.GenerateCoordsConfig(),
        equilibrate=[
            smee.mm.MinimizationConfig(),
            # short NPT equilibration simulation
            smee.mm.SimulationConfig(
                temperature=temperature,
                pressure=pressure,
                # n_steps=100_000,
                n_steps=10,
                timestep=2.0 * openmm.unit.femtosecond,
            ),
        ],
        production=smee.mm.SimulationConfig(
            temperature=temperature,
            pressure=pressure,
            # n_steps=1_000_000,
            n_steps=100,
            timestep=2.0 * openmm.unit.femtosecond,
        ),
        production_frequency=2000,
    )


def _vacuum_config(
    temperature: float, pressure: float | None
) -> descent.targets.thermo.SimulationConfig:
    """Return a default simulation configuration for simulations of the vacuum phase.

    Args:
        temperature: The temperature [K] at which to run the simulation.
        pressure: The pressure [atm] at which to run the simulation.

    Returns:
        The default simulation configuration.
    """
    temperature = temperature * openmm.unit.kelvin
    assert pressure is None

    return descent.targets.thermo.SimulationConfig(
        max_mols=1,
        gen_coords=smee.mm.GenerateCoordsConfig(),
        equilibrate=[
            smee.mm.MinimizationConfig(),
            smee.mm.SimulationConfig(
                temperature=temperature,
                pressure=None,
                # n_steps=50_000,
                n_steps=10,
                timestep=1.0 * openmm.unit.femtosecond,
            ),
        ],
        production=smee.mm.SimulationConfig(
            temperature=temperature,
            pressure=None,
            # n_steps=1_000_000,
            n_steps=100,
            timestep=1.0 * openmm.unit.femtosecond,
        ),
        production_frequency=500,
    )


# From https://github.com/SimonBoothroyd/descent/blob/7cd8062e2ff222047dfaccc6e45facf614abe9db/descent/targets/thermo.py#L385C1-L404C41
def default_config(
    phase: descent.targets.thermo.Phase, temperature: float, pressure: float | None
) -> descent.targets.thermo.SimulationConfig:
    """Return a default simulation configuration for the specified phase.

    Args:
        phase: The phase to return the default configuration for.
        temperature: The temperature [K] at which to run the simulation.
        pressure: The pressure [atm] at which to run the simulation.

    Returns:
        The default simulation configuration.
    """

    if phase.lower() == "bulk":
        return _bulk_config(temperature, pressure)
    elif phase.lower() == "vacuum":
        return descent.targets.thermo._vacuum_config(temperature, pressure)
    else:
        raise NotImplementedError(phase)


def run_simulation(
    phase: descent.targets.thermo.Phase,
    key: descent.targets.thermo.SimulationKey,
    system: smee.TensorSystem,
    force_field: smee.TensorForceField,
    output_dir: pathlib.Path,
) -> tuple[
    descent.targets.thermo.Phase, descent.targets.thermo.SimulationKey, pathlib.Path
]:
    """Run the given simulation and return the path to the frames from which the observables can be computed along with the phase and key."""
    import hashlib
    import pickle

    traj_hash = hashlib.sha256(pickle.dumps(key)).hexdigest()
    traj_name = f"{phase}-{traj_hash}-frames.msgpack"

    output_path = output_dir / traj_name

    if output_path.exists():
        logger.info(
            f"Simulation {phase} {key} already exists at {output_path}. Skipping."
        )

    else:
        config = default_config(phase, key.temperature, key.pressure)
        descent.targets.thermo._simulate(system, force_field, config, output_path)

    return (phase, key, output_path)


def run_required_simulations(
    trainable: Any,
    x: torch.Tensor,
    required_simulations: Dict[
        descent.targets.thermo.Phase, Dict[Any, smee.TensorSystem]
    ],
    output_dir: pathlib.Path,
    max_workers: int = 2,
) -> Dict[str, Dict[Any, Any]]:
    """
    Plan and run the minimum number of required simulations in parallel.

    Args:
        trainable: The trainable object for force field parameters.
        x: The current parameter tensor.
        required_simulations: Dictionary of required simulations, indexed by phase and key.
        output_dir: Directory to store simulation outputs.
        max_workers: Number of parallel workers.
        mp_context: Multiprocessing context (optional).

    Returns:
        frames: Nested dict of simulation results, indexed by phase and key.
    """
    from multiprocessing import get_context

    mp_context = get_context("spawn")

    sim_ff = trainable.to_force_field(x.detach().clone())
    frames: Dict[str, Dict[Any, Any]] = {
        phase: {} for phase in required_simulations.keys()
    }
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=mp_context) as pool:
        simulations = []
        for phase, systems in required_simulations.items():
            for key, system in systems.items():
                simulations.append(
                    pool.submit(
                        run_simulation,
                        **{
                            "phase": phase,
                            "key": key,
                            "system": system,
                            "force_field": sim_ff,
                            "output_dir": output_dir,
                        },
                    )
                )
        for job in tqdm.tqdm(
            as_completed(simulations),
            desc="Running simulations",
            total=len(simulations),
        ):
            phase, key, sim_path = job.result()
            frames[phase][key] = sim_path
    return frames


def iter_predicted_properties(
    entries: list[Any],
    entry_to_simulation: list[Any],
    required_simulations: dict,
    frames: dict,
    trainable: Any,
    x: torch.Tensor,
) -> Iterator[Tuple[Any, Any, Any]]:
    """
    Iterate over entries and yield the predicted condensed phase
    properties and their standard deviations.

    Args:
        entries: List of dataset entries.
        entry_to_simulation: List of simulation keys for each entry.
        per_type_scales: Dict of scaling factors by type.
        required_simulations: Dict of required simulation systems.
        frames: Dict of simulation results.
        trainable: The trainable object for force field parameters.
        x: The current parameter tensor.

    Yields:
        Tuple of (entry, predcted_prop, std) for each entry.
    """
    from collections import defaultdict

    import openmm.unit

    # remake the force field to make sure the graident is correctly attached to the tensors
    force_field = trainable.to_force_field(x)
    for entry, keys in zip(entries, entry_to_simulation, strict=True):
        observables = defaultdict(dict)
        for sim_key in keys.values():
            temperature = sim_key.temperature * openmm.unit.kelvin
            pressure = (
                None
                if sim_key.pressure is None
                else sim_key.pressure * openmm.unit.atmospheres
            )
            obs = descent.targets.thermo._Observables(
                *smee.mm.compute_ensemble_averages(
                    system=required_simulations["bulk"][sim_key],
                    force_field=force_field,
                    frames_path=frames["bulk"][sim_key],
                    temperature=temperature,
                    pressure=pressure,
                ),
            )
            observables["bulk"][sim_key] = obs

        pred, std = descent.targets.thermo._predict(
            entry=entry,
            keys=keys,
            observables=observables,
            systems=required_simulations,
        )
        yield entry, pred, std
