"""General trainer script with batched closure for dimers, energies/forces and smart batching for liquids which can easily be extended to fit to combinations
of targets.
Note uses the LM optimiser exclusively to avoid lots of liquid evaluations.
"""

import copy
import functools
import logging
import os
import pathlib
import pprint

import datasets
import descent.optim
import descent.targets
import descent.targets.dimers
import descent.targets.thermo
import descent.train
import descent.utils.loss
import descent.utils.molecule
import descent.utils.reporting
import loguru
import smee
import smee.converters
import smee.utils
import torch
import tqdm
from torch.utils.tensorboard import SummaryWriter

from .convert import convert_to_offxml, to_vdw_only_ff
from .models import WorkflowConfig

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


logger = loguru.logger


def report(
    step,
    x: torch.Tensor,
    loss,
    gradient,
    hessian,
    step_quality,
    accept_step,
    trainable: descent.train.Trainable,
    topologies: dict[str, smee.TensorTopology],
    writer: SummaryWriter,
):
    # Log to console
    logging.info(
        f"step: {step} "
        f"loss: {loss.detach().item():.5f} "
        f"quality: {step_quality.detach().item():.5f} "
        f"accept: {str(accept_step).lower()}"
    )
    logging.info(f"x: {x.detach().cpu().numpy()}")

    # Log to TensorBoard
    writer.add_scalar("Loss", loss.detach().item(), step)
    writer.add_scalar("Step Quality", step_quality.detach().item(), step)
    writer.add_scalar("Accept Step", int(accept_step), step)

    # Optionally log the parameters (x) as histograms
    writer.add_histogram("Parameters", x.detach().cpu().numpy(), step)

    # Optionally log gradients if available
    if gradient is not None:
        writer.add_histogram("Gradients", gradient.detach().cpu().numpy(), step)

    # Optionally log the Hessian if available
    if hessian is not None:
        writer.add_histogram("Hessian", hessian.detach().cpu().numpy(), step)

    # if accept_step:
    #     ff = trainable.to_force_field(x.detach().clone().requires_grad_(False))
    #
    #     with tempfile.TemporaryDirectory() as tmp_dir:
    #         descent.targets.thermo.predict(
    #             thermo_dataset, ff, topologies, pathlib.Path(tmp_dir), None, None, True
    #         )


def default_dimer_closure(
    trainable: "descent.train.Trainable",
    topologies: dict[str, smee.TensorTopology],
    dataset: datasets.Dataset,
    batch_size: int = 1,
) -> descent.optim.ClosureFn:
    """Return a default closure function for training against thermodynamic
    properties.

    Args:
        trainable: The wrapper around trainable parameters.
        topologies: The topologies of the molecules present in the dataset, with keys
            of mapped SMILES patterns.
        dataset: The dataset to train against.
        batch_size: The number of dimer entries to calculate the gradient and hessian for in each batch, gradients and hessian will be averaged over the batch.

    Returns:
        The default closure function.
    """
    import math

    import more_itertools
    import tqdm

    def closure_fn(
        x: torch.Tensor,
        compute_gradient: bool,
        compute_hessian: bool,
    ):
        total_loss, grad, hess = (
            torch.zeros(size=(1,), device=x.device.type),
            None,
            None,
        )
        # get the total number of dimers and configs to get the RMSE and average gradient and hessian
        n_dimers = len(dataset)
        total_points = sum([len(d["energy"]) for d in dataset])

        for batch_ids in tqdm.tqdm(
            more_itertools.batched([i for i in range(n_dimers)], batch_size),
            desc="Calculating dimers",
            ncols=80,
            total=math.ceil(n_dimers / batch_size),
        ):
            batch = dataset.select(indices=batch_ids)
            actuall_batch_size = len(batch)
            batch_configs = sum([len(d["energy"]) for d in batch])

            def loss_fn(_x):
                ff_vdw = to_vdw_only_ff(trainable.to_force_field(_x))
                y_ref, y_pred = descent.targets.dimers.predict(
                    batch, ff_vdw, topologies
                )
                return torch.sqrt(((y_pred - y_ref) ** 2).mean())

            # def loss_fn(_x):
            #     ff_vdw = to_vdw_only_ff(trainable.to_force_field(_x))
            #     y_ref, y_pred = descent.targets.dimers.predict(
            #         batch, ff_vdw, topologies
            #     )
            #     # Weight the loss by the Boltzmann factor of the QM energy
            #     qm_0 = min(y_ref)
            #     weights = torch.exp(-(y_ref - qm_0) / 0.59)  # kT at 300K
            #     weights /= weights.sum()
            #     weighted_sq_diff = torch.square(y_pred - y_ref) * weights

            #     return torch.sqrt(weighted_sq_diff).mean()

            loss = loss_fn(x)

            if compute_hessian:
                hessian = torch.autograd.functional.hessian(
                    loss_fn, x, vectorize=True, create_graph=False
                ).detach()
                if hess is None:
                    hess = hessian * actuall_batch_size
                else:
                    hess += hessian * actuall_batch_size
            if compute_gradient:
                (gradient,) = torch.autograd.grad(loss, x, create_graph=False)
                gradient = gradient.detach()
                if grad is None:
                    grad = gradient * actuall_batch_size
                else:
                    grad += gradient * actuall_batch_size

            # we want the overal rmse for reporting
            total_loss += torch.square(loss.detach()) * batch_configs

        final_grad = grad / n_dimers if grad is not None else None
        final_hess = hess / n_dimers if hess is not None else None

        return torch.sqrt(total_loss / total_points), final_grad, final_hess

    return closure_fn


def run_simulation(
    phase: descent.targets.thermo.Phase,
    key: descent.targets.thermo.SimulationKey,
    system: smee.TensorSystem,
    force_field: smee.TensorForceField,
    output_dir: pathlib.Path,
) -> tuple[str, str, str]:
    """Run the given simulation and return the path to the frames from which the observables can be computed along with the phase and key."""
    import hashlib
    import pickle

    traj_hash = hashlib.sha256(pickle.dumps(key)).hexdigest()
    traj_name = f"{phase}-{traj_hash}-frames.msgpack"

    output_path = output_dir / traj_name

    config = descent.targets.thermo.default_config(phase, key.temperature, key.pressure)
    descent.targets.thermo._simulate(system, force_field, config, output_path)
    return (phase, key, output_path)
    # return output_path


def smart_liquid_closure(
    trainable: "descent.train.Trainable",
    topologies: dict[str, smee.TensorTopology],
    dataset: datasets.Dataset,
    output_dir: pathlib.Path,
    per_type_scales: dict[descent.targets.thermo.DataType, float] | None = None,
) -> descent.optim.ClosureFn:
    """Return a default closure function for training against thermodynamic
    properties.

    Notes:
        The closure computes the properties in batches of size one to reduce the memory footprint.
        The liquid simulations are deduplicated where possible.

    Args:
        trainable: The wrapper around trainable parameters.
        topologies: The topologies of the molecules present in the dataset, with keys
            of mapped SMILES patterns.
        dataset: The dataset to train against.
        per_type_scales: The scale factor to apply to each data type.
        verbose: Whether to log additional information about predictions.

    Returns:
        The default closure function.
    """

    def closure_fn(
        x: torch.Tensor,
        compute_gradient: bool,
        compute_hessian: bool,
    ):
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from multiprocessing import get_context

        import openmm.unit

        total_loss, grad, hess = (
            torch.zeros(size=(1,), device=x.device.type),
            None,
            None,
        )

        entries = [*descent.utils.dataset.iter_dataset(dataset)]
        # plan the minimum number of required simulations
        required_simulations, entry_to_simulation = (
            descent.targets.thermo._plan_simulations(entries, topologies)
        )
        # run the simulations and store the path to the simulation data to be used later
        # detach the tensor to pass through the pool, only used for the simulation the attched tensor is used for the gradient later
        sim_ff = trainable.to_force_field(x.detach().clone())
        frames = {phase: {} for phase in required_simulations.keys()}
        with ProcessPoolExecutor(
            max_workers=2, mp_context=get_context("spawn")
        ) as pool:
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

        # As above, but sequential with no multiprocessing
        # for phase, systems in tqdm.tqdm(
        #     required_simulations.items(),
        #     desc="Running simulations",
        #     ncols=80,
        #     total=len(required_simulations),
        # ):
        #     print(f"Running {phase} simulations")
        #     for key, system in systems.items():
        #         print(f"Running {key}, {system}")
        #         frames[phase][key] = run_simulation(
        #             phase, key, system, sim_ff, output_dir
        #         )

        # frames = {
        #     phase: {
        #         key: run_simulation(phase, key, system, force_field, output_dir)
        #         for key, system in systems.items()
        #     }
        #     for phase, systems in required_simulations.items()
        # }

        # remake the force field to make sure the graident is correctly attached to the tensors
        force_field = trainable.to_force_field(x)
        # load each of the set of frames and calculate the loss
        for entry, keys in tqdm.tqdm(
            zip(entries, entry_to_simulation, strict=True),
            desc="Calculating observables",
            ncols=80,
            total=len(entries),
        ):
            type_scale = per_type_scales.get(entry["type"], 1.0)
            ref = entry["value"] * type_scale
            # gather the observables for this entry
            from collections import defaultdict

            observables = defaultdict(dict)
            predicted = []
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
            # print(observables)
            pred, _ = descent.targets.thermo._predict(
                entry=entry,
                keys=keys,
                observables=observables,
                systems=required_simulations,
            )
            predicted.append(pred * type_scale)
            y_pred = torch.stack(predicted)
            print(y_pred)
            y_ref = smee.utils.tensor_like(ref, y_pred)

            loss = (y_pred - y_ref) ** 2

            if compute_hessian:
                print(x)
                print(y_pred)
                hessian = descent.utils.loss.approximate_hessian(x, y_pred).detach()
                if hess is None:
                    hess = hessian
                else:
                    hess += hessian
            if compute_gradient:
                gradient = torch.autograd.grad(loss, x, retain_graph=True)[0].detach()
                if grad is None:
                    grad = gradient
                else:
                    grad += gradient

            total_loss += loss.detach()
            # clear the graph
            torch.cuda.empty_cache()

        final_grad = grad / len(dataset) if grad is not None else None
        final_hess = hess / len(dataset) if hess is not None else None

        return total_loss, final_grad, final_hess

    return closure_fn


def train(
    config: WorkflowConfig,
) -> None:
    """
    Train a force field to a mixture of dimer and thermo data, using the
    Levenberg-Marquardt algorithm.
    """

    # Make sure the fit directory exists
    config.fit_dir.mkdir(parents=True, exist_ok=True)

    # Save a copy of the config file
    config.to_file(config.fit_dir / "config.yaml")

    logger.info("Parameter Config:")
    logger.info(pprint.pprint(config.training.parameters))

    # load up the dataset options
    dimer_dataset = (
        datasets.load_from_disk(config.dimer.processed_data_dir)
        if config.dimer is not None
        else None
    )

    thermo_dataset = (
        datasets.load_from_disk(config.thermo.processed_data_dir)
        if config.thermo is not None
        else None
    )

    # May add in future...
    # energy_dataset = None

    logger.info("Loading initial force field and topologies")
    ff_initial, topologies = torch.load(config.pt_ff_and_tops_path, map_location="cuda")

    # edit the water assignment matrix to constrain the charges

    # water_top = topologies["[O:1]([H:2])[H:3]"]
    # we need to set both hydrogens to the same charge parameter, and the vsite to -2 times it
    # print(water_top.parameters['Electrostatics'].assignment_matrix)
    # set the oxygen parameter to not be used
    # water_top.parameters["Electrostatics"].assignment_matrix[0] = water_top.parameters["Electrostatics"].assignment_matrix[0] * 0.0
    # set the hydrogens to be the same
    # water_top.parameters['Electrostatics'].assignment_matrix[2] = water_top.parameters['Electrostatics'].assignment_matrix[1]
    # set the vsite
    # water_top.parameters['Electrostatics'].assignment_matrix[3] = water_top.parameters['Electrostatics'].assignment_matrix[1] * -2.0
    # print(water_top.parameters['Electrostatics'].assignment_matrix)
    # print(water_top.parameters['Electrostatics'].assignment_matrix @ ff_initial.potentials_by_type["Electrostatics"].parameters)
    # print(ff_initial.v_sites)
    # make sure all tensors on the GPU
    # water_top.to("cuda")

    # print vdW
    logger.info("Initial Force Field Summary:")
    descent.utils.reporting.print_potential_summary(
        ff_initial.potentials_by_type["vdW"]
    )
    # # print Electro
    # descent.utils.reporting.print_potential_summary(
    #     ff_initial.potentials_by_type["Electrostatics"]
    # )

    # Edit the param config to match the initial force field
    parameters = config.training.parameters

    for k, v in parameters.items():
        for param_type in ["include", "exclude"]:
            if param_type in v:
                parameters[k][param_type] = [
                    p
                    for p in ff_initial.potentials_by_type[k].parameter_keys
                    if p.id in v[param_type]
                ]

    logger.info("Parameters after editing to match initial force field:")
    logger.info(pprint.pprint(parameters))

    trainable = descent.train.Trainable(
        copy.deepcopy(ff_initial),
        parameters={
            k: descent.train.ParameterConfig(**v) for k, v in parameters.items()
        },
        attributes={
            k: descent.train.AttributeConfig(**v)
            for k, v in config.training.attributes.items()
        },
    )
    # build the combined closure
    logging.info("Creating closure function")
    closures_to_combine = {}
    if thermo_dataset is not None:
        liquid_dir = config.fit_dir.joinpath("liquid-cache")
        liquid_dir.mkdir(parents=True, exist_ok=True)
        closures_to_combine["thermo"] = smart_liquid_closure(
            trainable=trainable,
            topologies=topologies,
            dataset=thermo_dataset,
            per_type_scales=config.thermo_scales,
            output_dir=liquid_dir,
        )
    if dimer_dataset is not None:
        closures_to_combine["dimer"] = default_dimer_closure(
            trainable=trainable,
            topologies=topologies,
            dataset=dimer_dataset,
            batch_size=100,
        )
    if len(closures_to_combine) > 1:
        closure_fn = descent.utils.loss.combine_closures(
            closures_to_combine,
            weights={
                target: config.training.weights[target]
                for target in closures_to_combine
            },
            verbose=True,
        )
    else:
        closure_fn = list(closures_to_combine.values())[0]
    correct_fn = trainable.clamp

    lm_config = descent.optim.LevenbergMarquardtConfig(
        mode="adaptive", n_convergence_criteria=0, max_steps=10
    )
    with SummaryWriter(config.fit_dir / "tensorboard") as writer:
        report_fn = functools.partial(
            report,
            trainable=trainable,
            topologies=topologies,
            writer=writer,
        )

        x_final = descent.optim.levenberg_marquardt(
            trainable.to_values(), lm_config, closure_fn, correct_fn, report_fn
        )

    ff_final = trainable.to_force_field(x_final)

    logger.info("Final Force Field Summary:")
    descent.utils.reporting.print_potential_summary(ff_final.potentials_by_type["vdW"])
    # descent.utils.reporting.print_potential_summary(
    #     ff_final.potentials_by_type["Electrostatics"]
    # )

    torch.save(ff_final, config.fit_dir.joinpath("final_ff.pt"))

    # Save the final offxml force field
    convert_to_offxml(
        config.training.starting_force_field_path,
        ff_final,
        config.output_ff_path,
    )

    # Also save a description of the fit with the output force field
    config.output_description_path.write_text(config.experiment_description)

    # Save a report on the dimer energies
    if dimer_dataset is not None:
        tops = {smiles: topology.to("cpu") for smiles, topology in topologies.items()}
        descent.targets.dimers.report(
            dimer_dataset,
            {
                "LJ Initial": to_vdw_only_ff(ff_initial).to("cpu"),
                "LJ Opt": to_vdw_only_ff(ff_final).to("cpu"),
            },
            {"LJ Initial": tops, "LJ Opt": tops},
            config.fit_dir / "energies.html",
        )
