"""Time-dependent VMC (t-VMC) of the TFIM quantum-anneal quench.

Evolves the same quench as ``exact_vs_dmrg.ipynb`` (schedule ``H(s) = pi*t_a*[J(s)
Hzz + Gamma(s) Hx]``, ``s: 0 -> 1``) variationally, using the blurred TDVP
driver in :mod:`src.tdvp_blurred`. The time-dependent Hamiltonian is a single
weighted Ising operator (:class:`src.operator.SpinGlassIsing`) rebuilt at each time
and handed to the driver as its generator.

This script runs the simulation and checkpoints the state; it does NOT compute
correlations (use ``verify_tvmc.py`` for that, off the saved checkpoint).

Usage:
    python run_tvmc.py --config configs/tvmc_3x3.yaml
"""

import argparse
import hashlib
import json
import os
from functools import partial

import numpy as np
import yaml

import netket as nk

import jax
import optax
from netket.experimental.dynamics import RK45, Heun, Euler

from src.exact import load_instance, coupling_matrix, schedule_interpolators
from src.operator import SpinGlassIsing
from src.tdvp_blurred import TDVPBlurred
from src.logger import CheckpointCallback
from src.utils import get_save_path, enumerate_cycles
from src.models import CorrelatorJastrow, RBM, JastrowRBM, JastrowPlaquette


def build_generator(config):
    """Return ``(hilbert, H)`` for the quench.

    ``H(s)`` is the time-dependent generator the TDVP driver consumes.
    """
    weights, edges = load_instance(
        config.get("instance_dir", "./data_dwave/instances"),
        config["topology"],
        config["shape"],
        config["instance"],
        precision=config["precision"],
    )
    L = max(max(i, j) for i, j in edges) + 1
    Jmat = coupling_matrix(L, weights, edges)
    hi = nk.hilbert.Spin(0.5, N=L)
    graph = nk.graph.Graph(edges=[list(e) for e in edges])
    _, f_gamma, f_J = schedule_interpolators(config["schedule"])

    pref = 2 * np.pi * config["t_a"]

    def H(s, scale=True):
        if scale:
            return SpinGlassIsing(
                hi,
                graph,
                h=pref * float(f_gamma(s)),
                J=pref * float(f_J(s)),
                Jmat=Jmat,
            )
        else:
            return SpinGlassIsing(
                hi,
                graph,
                h=float(f_gamma(s)),
                J=float(f_J(s)),
                Jmat=Jmat,
            )

    return hi, H


def build_sampler(config, hi):
    """Build the Metropolis sampler for the given hilbert space."""
    L = hi.size
    # Number of MC sweeps between samples; defaults to the system size N = Lx*Ly.
    sweep_size = config.get("sweep_size") or L
    kind = config.get("sampler", "local")
    if kind == "local":
        return nk.sampler.MetropolisLocal(
            hi, n_chains=config["n_chains"], sweep_size=sweep_size
        )
    elif kind == "pt":
        # Parallel tempering with local single-spin proposals: evolves `n_replicas`
        # replicas per physical chain along a beta ladder (beta=1 physical) and swaps
        # between them.
        return nk.sampler.ParallelTemperingLocal(
            hi,
            n_replicas=config.get("pt_n_replicas", 16),
            betas=config.get("pt_betas", "linear"),
            n_chains=config["n_chains"],
            sweep_size=sweep_size,
        )
    elif kind == "exact":
        # Exact i.i.d. sampling from the model's Born distribution.
        return nk.sampler.ExactSampler(hi)
    raise ValueError(f"unknown sampler {kind}")


def build_state(config, hi):
    sampler = build_sampler(config, hi)
    model_name = config.get("model", "jastrow")
    if model_name == "jastrow":
        model = CorrelatorJastrow(
            orders=tuple(config.get("orders", (2, 4))),
            rank=config.get("rank", 1),
            legacy_rank_k2=config.get("legacy_rank_k2", True),
            param_dtype=complex,
            param_initializer=jax.nn.initializers.normal(0.01 / hi.size),
            compute_dtype=config.get("compute_dtype"),
        )
    elif model_name == "rbm":
        model = RBM(
            alpha=config.get("alpha", 1),
            param_dtype=complex,
            compute_dtype=config.get("compute_dtype"),
        )
    elif model_name == "jastrow_rbm":
        model = JastrowRBM(
            orders=tuple(config.get("orders", (2, 4))),
            rank=config.get("rank", 1),
            legacy_rank_k2=config.get("legacy_rank_k2", True),
            alpha=config.get("alpha", 1),
            param_dtype=complex,
            param_initializer=jax.nn.initializers.normal(0.01 / hi.size),
            compute_dtype=config.get("compute_dtype"),
        )
    elif model_name == "jastrow_plaquette":
        # Graph-native cycle-body term on the instance's `cycle_size`-cycles, in addition
        # to the democratic [orders] Jastrow. Cycles derive from the instance edge list
        # (same load as build_generator), so they are fixed by topology/shape/instance.
        # cycle_size=4 -> plaquettes (default); use 6 for a girth-6 graph like diamond.
        cycle_size = config.get("cycle_size", 4)
        _, edges = load_instance(
            config.get("instance_dir", "./data_dwave/instances"),
            config["topology"],
            config["shape"],
            config["instance"],
            precision=config["precision"],
        )
        plaquettes = enumerate_cycles(edges, cycle_size)
        model = JastrowPlaquette(
            orders=tuple(config.get("orders", (2, 4))),
            rank=config.get("rank", 1),
            legacy_rank_k2=config.get("legacy_rank_k2", True),
            plaquettes=plaquettes,
            param_dtype=complex,
            param_initializer=jax.nn.initializers.normal(0.01 / hi.size),
            compute_dtype=config.get("compute_dtype"),
        )
    else:
        raise ValueError(
            f"unknown model {model_name}; only 'jastrow', 'rbm', "
            "'jastrow_rbm' and 'jastrow_plaquette' are supported"
        )
    return nk.vqs.MCState(
        sampler, model, n_samples=config["n_samples"], seed=config["seed"]
    )


def make_integrator(config, dt0=None):
    """Build the ODE integrator. ``dt0`` overrides the initial step (used on resume)."""
    name = config.get("integrator", "rk45")
    if name == "rk45":
        dt_min, dt_max = config.get("dt_min", 1e-4), config.get("dt_max", 1e-2)
        return RK45(
            dt_min if dt0 is None else dt0,
            adaptive=True,
            rtol=1e-5,
            dt_limits=(dt_min, dt_max),
        )
    elif name == "heun":
        dt = config.get("dt", config.get("dt_max", 1e-2))
        return Heun(dt if dt0 is None else dt0)
    elif name == "euler":
        dt = config.get("dt", config.get("dt_max", 1e-2))
        return Euler(dt if dt0 is None else dt0)
    raise ValueError(f"unknown integrator {name}")


def init_wandb(config):
    if not config.get("wandb", False):
        return None
    import wandb

    run_id = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return wandb.init(
        dir=os.environ.get("WANDB_DIR", "."),
        entity="rwiersema-",
        project="dwave_nqs",
        config=config,
        id=run_id,
        resume="allow",
        group=f"DWAVE",
        mode=None if jax.process_index() == 0 else "disabled",
    )


def v_score_callback(step, log, driver, *, threshold: float = 1e-4):
    """Callback for early stopping"""
    e = driver._loss_stats
    E, var = float(e.mean.real), float(e.variance)
    vscore = driver.state.hilbert.size * var / E**2
    return vscore > threshold


def main(config):
    save_path = get_save_path(config)
    print(f"Save path: {save_path}")

    hi, H = build_generator(config)
    vstate = build_state(config, hi)
    print(f"Sampler: {config.get('sampler', 'local')}")
    print(f"Number of parameters: {vstate.n_parameters}")

    wandb_run = init_wandb(config)
    save_times = np.linspace(0.0, 1.0, config.get("n_save_times", 21))
    ckpt = CheckpointCallback(
        save_path,
        logger=nk.logging.RuntimeLog(),
        every_n_steps=config.get("every_n_steps", 25),
        wandb_run=wandb_run,
        save_times=save_times,
    )

    logger, restored = ckpt.restore_logger()
    if ckpt.done:
        print("t-VMC already complete for this config.")
        return

    if restored and (state := ckpt.restore_state()) is not None:
        vstate = state
        want_legacy = config.get("legacy_rank_k2", True)
        got_legacy = getattr(vstate.model, "legacy_rank_k2", True)
        if got_legacy != want_legacy:
            raise ValueError(
                f"checkpoint in {save_path} was written with legacy_rank_k2={got_legacy}, "
                f"but this config asks for {want_legacy}; refusing to resume across "
                "parameterizations"
            )
        # ``t`` is the anneal parameter s; ``step`` is the driver's step counter
        # (the History "iters" of t is s itself, not the step count).
        t0 = float(np.ravel(logger["t"].to_dict()["value"])[-1])
        try:
            step = int(np.ravel(logger["step"].to_dict()["value"])[-1])
        except (KeyError, IndexError):
            step = 0
        # Restore the last adaptive dt so stepping resumes where it left off.
        try:
            dt0 = float(np.ravel(logger["dt"].to_dict()["value"])[-1])
        except (KeyError, IndexError):
            dt0 = None
        print(f"Resuming from s = {t0:.4f} (step {step}, dt = {dt0})")
        for _ in range(100):
            vstate.reset()
            vstate.sample()
    else:
        # Fresh start: prepare the initial state = ground state of H(s=0).
        print("Preparing initial state: VMC ground state of H(s=0)...")
        lr_schedule = optax.cosine_decay_schedule(
            config["vmc_lr"], config["n_vmc_steps"], alpha=config["vmc_lr"] / 10
        )
        gs_driver = nk.driver.VMC_SR(
            hamiltonian=H(0.0, scale=False),
            optimizer=optax.sgd(lr_schedule),
            variational_state=vstate,
            diag_shift=config["diag_shift"],
        )
        gs_driver.run(
            n_iter=config["n_vmc_steps"],
            callback=partial(v_score_callback, threshold=1e-4),
        )
        e0 = gs_driver._loss_stats
        E, var = float(e0.mean.real), float(e0.variance)
        vscore = hi.size * var / E**2
        gs_stats = {
            "energy_real": E,
            "energy_imag": float(e0.mean.imag),
            "variance": var,
            "error_of_mean": float(e0.error_of_mean),
            "vscore": vscore,
            "rel_std": float(np.sqrt(var)) / abs(E),
        }
        with open(os.path.join(save_path, "gs_energy.json"), "w") as f:
            json.dump(gs_stats, f, indent=2)
        print(f"GS prep done: E = {E:.6f}, Var = {var:.4e}, V-score = {vscore:.3e}")
        if wandb_run is not None:
            wandb_run.summary.update({f"gs_{k}": v for k, v in gs_stats.items()})
        # Abort if the initial state is not converged:
        if gs_driver.step_count == config["n_vmc_steps"]:
            raise RuntimeError(
                f"H(0) ground-state prep did not converge: V-score {vscore:.3e}"
            )
        t0, step, dt0 = 0.0, 0, None

    integrator = make_integrator(config, dt0=dt0)
    driver = TDVPBlurred(
        H,
        vstate,
        integrator,
        t0=t0,
        q=config.get("q", 0.0),
        rcond=config["rcond"],
        rcond_smooth=config["rcond_smooth"],
        snr_atol=config.get("snr_atol"),
        holomorphic=True,
        distributed_eigh=config.get("distributed_eigh", False),
        cache_within_step=config.get("cache_within_step", True),
        error_dump_dir=save_path,
    )
    driver._step_count = step

    callbacks = [ckpt]

    print(f"Running t-VMC from s={t0:.4f} to s=1.0 ...")
    driver.run(
        1.0 - t0,
        out=logger,
        callback=callbacks,
        show_progress=True,
    )
    ckpt.finish(driver.state)
    print("t-VMC done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t-VMC quench run")
    parser.add_argument("--config", required=True, help="path to YAML config")
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    main(config)
