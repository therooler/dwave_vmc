"""Verify a finished t-VMC run against DMRG.

Loads a checkpointed variational state produced by ``run_tvmc.py`` (off disk, so
correlations are computed once and not during the run), evaluates the 2-local ZZ
correlations, and compares them to the DMRG reference. By default the final state is
used; pass ``--t`` to restore a per-time snapshot instead (e.g. ``--t 0.6``).

Usage:
    python verify_tvmc.py --config configs/tvmc_3x3.yaml
    python verify_tvmc.py --config configs/tvmc_3x3.yaml --t 0.6
"""

import argparse
import os

import numpy as np
import yaml

import netket as nk

from src.dmrg import load_mps_corrs, load_qpu_corrs, correlation_error
from src.logger import CheckpointCallback
from src.utils import get_save_path


def tvmc_correlations(vstate):
    """Flat arrays ``(means, errors)`` of <sigma^z_i sigma^z_j>, i < j, from a
    variational state. ``errors`` is the Monte Carlo standard error of the mean
    on each correlation estimate."""
    hi = vstate.hilbert
    L = hi.size
    means, errors = [], []
    for i in range(L):
        for j in range(i + 1, L):
            op = nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmaz(hi, j)
            stats = vstate.expect(op)
            means.append(stats.mean.real)
            errors.append(np.real(stats.error_of_mean))
    return np.array(means), np.array(errors)


def correlation_error_uncertainty(X, Xerr, Y):
    """1-sigma uncertainty on ``correlation_error(X, Y)`` propagated from the MC
    errorbars ``Xerr`` on ``X``. The reference ``Y`` is treated as exact.

    ``eps_c = sqrt(mean((X-Y)^2)) / sqrt(mean(Y^2))``; linear (delta-method)
    propagation gives ``d eps_c / d X_i = (X_i - Y_i) / (P * num * den)``.
    """
    X, Xerr, Y = np.asarray(X), np.asarray(Xerr), np.asarray(Y)
    P = X.size
    num = np.sqrt(np.mean((X - Y) ** 2))
    den = np.sqrt(np.mean(Y ** 2))
    if num == 0 or den == 0:
        return np.nan
    grad = (X - Y) / (P * num * den)  # d eps_c / d X_i
    return float(np.sqrt(np.sum((grad * Xerr) ** 2)))


def main(config, restore_time=None, n_samples=2**20, chunk_size=None):
    save_path = get_save_path(config, create=False)
    print(f"Save path: {save_path}")

    s = 1.0 if restore_time is None else float(restore_time)

    # Name carries the sample count and the snapshot time s. Exit early if it exists.
    out_path = os.path.join(save_path, f"corrs_n{n_samples}_t{s:1.3f}.npz")
    if os.path.exists(out_path):
        print(f"Correlations already saved, exiting: {out_path}")
        return

    ckpt = CheckpointCallback(save_path, logger=nk.logging.RuntimeLog(),
                              every_n_steps=1)
    # restore_time=None -> final state; otherwise the per-time snapshot saved by run_tvmc.
    if restore_time is None:
        vstate = ckpt.restore_state()
    else:
        vstate = ckpt.restore_state(f"/state_t{s:1.3f}.nk")
    if vstate is None:
        raise FileNotFoundError(
            f"No checkpoint state found in {save_path}"
            + ("" if restore_time is None else f" for t={s:1.3f}")
        )

    topology, shape, instance, t_a = (
        config["topology"], config["shape"], config["instance"], config["t_a"]
    )
    vstate.n_samples = n_samples
    if chunk_size is not None:
        vstate.chunk_size = chunk_size
    corrs_tvmc, corrs_err = tvmc_correlations(vstate)

    print(f"\n{topology} {tuple(shape)}  instance {instance}  t_a={t_a}ns  s={s:.3f}")

    # Exact MPS reference, stored per topology/shape and bond dim (chi).
    err_mps = err_mps_std = None
    try:
        corr_dir = config.get("corr_dir", "./data_dwave/correlations/mps")
        mps = load_mps_corrs(corr_dir, topology, shape, t_a, config["precision"])[instance]
        err_mps = correlation_error(corrs_tvmc, mps)[0]
        err_mps_std = correlation_error_uncertainty(corrs_tvmc, corrs_err, mps)
        print(f"  rel. error  t-VMC vs MPS : {err_mps:.4f} +/- {err_mps_std:.4f}")
    except FileNotFoundError:
        print("  (no MPS reference for this instance/t_a)")

    # D-Wave QPU reference, same per topology/shape layout but under a
    # per-processor-generation folder (adv1/adv2) and no chi sub-dir.
    err_qpu = err_qpu_std = None
    try:
        qpu_dir = config.get("qpu_dir", "./data_dwave/correlations/qpu")
        qpu = load_qpu_corrs(
            qpu_dir, topology, shape, t_a, config["precision"],
            processor=config.get("qpu_processor"),
        )[instance]
        err_qpu = correlation_error(corrs_tvmc, qpu)[0]
        err_qpu_std = correlation_error_uncertainty(corrs_tvmc, corrs_err, qpu)
        print(f"  rel. error  t-VMC vs QPU : {err_qpu:.4f} +/- {err_qpu_std:.4f}")
    except FileNotFoundError:
        print("  (no QPU reference for this instance/t_a)")

    np.savez(
        out_path,
        correlations=corrs_tvmc,
        correlation_errorbars=corrs_err,  # MC standard error of the mean, per pair
        # `correlation_error` kept as the MPS error for backward compatibility.
        correlation_error=np.nan if err_mps is None else err_mps,
        correlation_error_mps=np.nan if err_mps is None else err_mps,
        correlation_error_mps_std=np.nan if err_mps_std is None else err_mps_std,
        correlation_error_qpu=np.nan if err_qpu is None else err_qpu,
        correlation_error_qpu_std=np.nan if err_qpu_std is None else err_qpu_std,
    )
    print(f"Saved correlations to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="verify a t-VMC run")
    parser.add_argument("--config", required=True, help="path to YAML config")
    parser.add_argument(
        "--t", type=float, default=None,
        help="anneal parameter s of the snapshot to restore (e.g. 0.6); "
             "default uses the final state",
    )
    parser.add_argument(
        "--n-samples", type=int, default=2**20,
        help="number of samples to set on the loaded vstate",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="chunk size to set on the loaded vstate (default: no chunking)",
    )
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    main(config, restore_time=args.t,
         n_samples=args.n_samples, chunk_size=args.chunk_size)
