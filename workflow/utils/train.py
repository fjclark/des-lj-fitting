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
from functools import partial

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
from openff.toolkit import ForceField
from torch.utils.tensorboard import SummaryWriter

from .convert import convert_to_offxml, to_vdw_only_ff
from .get_fn import get_fn
from .loss import LossFnProto
from .models import WorkflowConfig
from .plot import plot_loss, plot_vdw_parameter_changes
from .simulate import iter_predicted_properties, run_required_simulations

# Make the Descent (specifically the LM optimiser) logging more verbose
logging.basicConfig(
    level=logging.INFO,  # or logging.DEBUG for more detail
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logging.getLogger("descent").setLevel(logging.DEBUG)

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
    config: WorkflowConfig,
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

    # Plot the loss so we can quickly check without opening tensorboard
    plot_loss(
        [config],
        output_path=config.fit_dir / "loss_plot.png",
    )

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
    loss_fn: LossFnProto,
    batch_size: int = 1,
    vdw_only: bool = True,
) -> descent.optim.ClosureFn:
    """Return a default closure function for training against thermodynamic
    properties.

    Args:
        trainable: The wrapper around trainable parameters.
        topologies: The topologies of the molecules present in the dataset, with keys
            of mapped SMILES patterns.
        dataset: The dataset to train against.
        loss_fn: The loss function to use for training.
        batch_size: The number of dimer entries to calculate the gradient and hessian for in each batch, gradients and hessian will be averaged over the batch.
        vdw_only: Whether the reference energy is vdW only (or the total energy).

    Returns:
        The default closure function.
    """
    import math

    import more_itertools

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

            _loss_fn = partial(
                loss_fn,
                trainable=trainable,
                batch=batch,
                topologies=topologies,
                vdw_only=vdw_only,
            )

            loss = _loss_fn(x)

            if compute_hessian:
                hessian = torch.autograd.functional.hessian(
                    _loss_fn, x, vectorize=True, create_graph=False
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
        total_loss, grad, hess = (
            torch.zeros(size=(1,), device=x.device.type),
            None,
            None,
        )

        # Plan the required simulations
        entries = [*descent.utils.dataset.iter_dataset(dataset)]
        required_simulations, entry_to_simulation = (
            descent.targets.thermo._plan_simulations(entries, topologies)
        )

        # Run required simulations
        frames = run_required_simulations(
            trainable,
            x,
            required_simulations,
            output_dir,
            max_workers=2,
        )

        # Calculate the loss from the predicted properties
        for entry, pred, _ in iter_predicted_properties(
            entries,
            entry_to_simulation,
            required_simulations,
            frames,
            trainable,
            x,
        ):
            predicted = []
            type_scale = per_type_scales.get(entry["type"], 1.0)
            ref = entry["value"] * type_scale

            # Compute the loss for this entry
            predicted.append(pred * type_scale)
            y_pred = torch.stack(predicted)

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
    Train a force field to dimer and/or thermo data, using the
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
            loss_fn=get_fn(config.training.loss_fns["dimer"]),
            batch_size=100,
            vdw_only=config.dimer.vdw_only,
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
        mode="adaptive", n_convergence_criteria=0, max_steps=20
    )
    with SummaryWriter(config.fit_dir / "tensorboard") as writer:
        report_fn = functools.partial(
            report,
            trainable=trainable,
            topologies=topologies,
            writer=writer,
            config=config,
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

    # Plot the parameter changes from the initial force field
    plot_vdw_parameter_changes(
        original_ff=ForceField(config.training.starting_force_field_path),
        fitted_ff=ForceField(config.output_ff_path),
        output_path=config.fit_dir / "parameter_changes.png",
    )

    # Save a report on the dimer energies
    if dimer_dataset is not None:
        tops = {smiles: topology.to("cpu") for smiles, topology in topologies.items()}
        ff_initial = (
            to_vdw_only_ff(ff_initial).to("cpu")
            if config.dimer.vdw_only
            else ff_initial.to("cpu")
        )
        ff_final = (
            to_vdw_only_ff(ff_final).to("cpu")
            if config.dimer.vdw_only
            else ff_final.to("cpu")
        )
        descent.targets.dimers.report(
            dimer_dataset,
            {
                "LJ Initial": ff_initial,
                "LJ Opt": ff_final,
            },
            {"LJ Initial": tops, "LJ Opt": tops},
            config.fit_dir / "energies.html",
        )
