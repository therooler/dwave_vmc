"""Benchmark the QGT-inversion (S-matrix solve) scaling: ``distributed_eigh`` False vs True.

Times the *real* production solve ``src.tdvp_blurred._solve_dense`` on a real
``CorrelatorJastrow`` variational state, for a range of parameter counts ``n_params``,
comparing two modes:
  * ``single``   -- ``distributed_eigh=False`` (netket ``pinv_smooth`` -> ``jnp.linalg.eigh``)
  * ``sharded``  -- ``distributed_eigh=True`` (``jaxmg.syevd``) on a row-sharded QGT built
    map-side with ``sharded_to_dense`` (the full matrix is never materialized replicated).
    This is the production distributed path; ``_solve_dense`` asserts a pre-padded sharded Sd.
Records solve wall-time, writing one ``.npz`` per (size, mode) under
``data/eigh/ndev_<k>/`` (``k`` = devices used: single -> 1, sharded -> mesh size) for
plotting in ``paper_figures.ipynb`` (Figure 10).

Scaling knob: ``n_params = len(orders) * rank * N(N-1)/2`` for a Jastrow with all orders >= 2
(``src/models.py``).  With the default ``orders=[2,4]``, ``rank=1`` this is ``N(N-1)``, so we
sweep ``n_params`` by varying the number of spins ``N`` (``--sites``).

Process isolation: JAX's ``memory_stats()['peak_bytes_in_use']`` is a monotone high-water
mark with no public reset, and a single-device solve would contaminate the distributed
measurement.  So each (size, mode) runs in its own worker subprocess (re-exec of this file
with ``--worker``); the parent just orchestrates.  A worker that runs out of memory records
``oom=True`` and exits cleanly so the sweep continues.

Usage::

    # Full sweep on an 8-GPU box (single-device vs 8-GPU distributed):
    python benchmark_eigh.py

    # Restrict sizes / modes:
    python benchmark_eigh.py --sites 24,32 --modes single
    python benchmark_eigh.py --sites 64 --modes distributed

Output: ``data/eigh/ndev_{k}/solve_{mode}_np{n_params}.npz`` (``k`` = devices used;
schema in ``run_worker``).
"""

import argparse
import os
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
import statistics
import subprocess
import sys
import time


# --- defaults ----------------------------------------------------------------
# Lattice sizes N (spins).  n_params = len(orders)*rank*N(N-1)/2; for orders=[2,4], rank=1
# that is N(N-1): 40->1560, 64->4032, 90->8010, 128->16256, 160->25440, 200->39800,
# 240->57360.  The dense complex128 Sd is n_params^2 * 16 bytes (57360 -> ~52.6 GB), chosen
# to straddle the single-GPU memory wall while the 8-way sharded solve keeps going.
DEFAULT_SITES = [16*i for i in range(1, 20)]
DEFAULT_ORDERS = (2, 4)
DEFAULT_RANK = 1
DEFAULT_N_SAMPLES = 2048  # QGT-build sample count; build is NOT timed, only the solve is.
DEFAULT_RCOND = 1e-10
DEFAULT_RCOND_SMOOTH = 1e-7
DEFAULT_DIAG_SHIFT = 1e-3
SEED = 0
N_WARMUP = 2
N_REPS = 10


def n_params_for(n_sites, orders, rank, legacy_rank_k2=True):
    """Analytic Jastrow parameter count (all orders assumed >= 2).

    ``legacy_rank_k2=False`` drops the redundant rank axis on the k=2 block, which
    is then a single ``n_upper`` vector regardless of ``rank`` (see
    :class:`src.models.CorrelatorJastrow`).
    """
    n_upper = n_sites * (n_sites - 1) // 2
    total = 0
    for k in orders:
        if k == 1:
            total += n_sites
        elif k == 2 and not legacy_rank_k2:
            total += n_upper
        else:
            total += rank * n_upper
    return int(total)


def ndev_used(mode, n_devices_arg, n_visible):
    """Devices the solve actually runs on -> the ``ndev_<k>`` output subdir.

    ``single`` always uses one GPU; ``sharded`` uses ``n_devices_arg`` (0 = all visible),
    capped at what is visible. Parent and worker must agree here so the skip-check path
    matches the path the worker writes."""
    if mode == "single":
        return 1
    return n_visible if n_devices_arg in (0, None) else min(n_devices_arg, n_visible)


def _count_visible_devices():
    """``len(jax.local_devices())`` via a throwaway subprocess, so the parent never
    initializes JAX itself (which would grab GPU memory the workers need)."""
    proc = subprocess.run(
        [sys.executable, "-c", "import jax; print(len(jax.local_devices()))"],
        capture_output=True, text=True, check=False,
    )
    try:
        return int(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        print(f"[parent] could not count devices ({proc.stderr[:200]!r}); assuming 1")
        return 1


# ======================================================================================
# Worker: build one vstate, time one solve branch, save one npz.  Runs in a subprocess
# with device visibility / allocator env already set by the parent.
# ======================================================================================
def run_worker(args):
    # netket turns x64 on at import, so Sd is complex128 (matches production).
    import numpy as np
    import jax
    import jax.numpy as jnp
    import netket as nk

    from netket.optimizer.qgt.qgt_jacobian import QGTJacobian_DefaultConstructor
    from netket.utils.api_utils import partial_from_kwargs

    from src.models import CorrelatorJastrow
    from src.tdvp_blurred import _qgt_to_dense, _solve_dense, sharded_to_dense

    orders = tuple(int(o) for o in args.orders.split(","))
    N = args.sites
    n_params = n_params_for(N, orders, args.rank, args.legacy_rank_k2)
    devices = jax.local_devices()
    n_devices = len(devices)
    device_kind = devices[0].device_kind if devices else "cpu"
    # Devices actually used by the solve -> ndev_<k> subdir (single pins to 1; sharded
    # uses the mesh size). Keeps runs at different device counts separate on disk.
    ndev = ndev_used(args.mode, args.n_devices, n_devices)

    out_dir = os.path.join(args.out, f"ndev_{ndev}")
    out_path = os.path.join(out_dir, f"solve_{args.mode}_np{n_params}.npz")
    if os.path.exists(out_path) and not args.force:
        print(f"[worker] {out_path} exists, skipping (use --force to overwrite)")
        return

    print(
        f"[worker] mode={args.mode} N={N} orders={orders} rank={args.rank} "
        f"legacy_rank_k2={args.legacy_rank_k2} "
        f"n_params={n_params} n_devices={n_devices} ({device_kind}) "
        f"backend={jax.default_backend()}"
    )

    # Result record with OOM-safe defaults; filled in on success.
    rec = dict(
        mode=args.mode,
        n_params=n_params,
        n_sites=N,
        orders=np.asarray(orders),
        rank=args.rank,
        legacy_rank_k2=args.legacy_rank_k2,
        n_devices=ndev,
        n_samples=args.n_samples,
        t_min=np.nan,
        t_med=np.nan,
        t_mean=np.nan,
        t_std=np.nan,
        compile_time=np.nan,
        sd_bytes=n_params * n_params * 16,  # dense complex128 footprint
        oom=False,
        jax_version=jax.__version__,
        device_kind=device_kind,
    )

    try:
        # --- real variational state (mirrors run_tvmc.build_state) --------------------
        hi = nk.hilbert.Spin(0.5, N=N)
        model = CorrelatorJastrow(
            orders=orders,
            rank=args.rank,
            legacy_rank_k2=args.legacy_rank_k2,
            param_dtype=complex,
            param_initializer=jax.nn.initializers.normal(0.01 / hi.size),
        )
        sampler = nk.sampler.MetropolisLocal(hi, n_chains=min(256, args.n_samples))
        vstate = nk.vqs.MCState(sampler, model, n_samples=args.n_samples, seed=SEED)
        assert vstate.n_parameters == n_params, (
            f"n_parameters {vstate.n_parameters} != analytic {n_params}"
        )

        # --- real QGT (mirrors odefun_custom, src/tdvp_blurred.py:686-701) ------------
        samples_q = jax.block_until_ready(vstate.samples)
        pdf = jnp.ones(samples_q.shape[:-1])  # uniform weights (iw=ones), as in production
        S = partial_from_kwargs(
            QGTJacobian_DefaultConstructor,
            exclusive_arg_names=(("mode", "holomorphic")),
        )(
            vstate._apply_fun,
            vstate.parameters,
            vstate.model_state,
            samples_q,
            pdf=pdf / pdf.size,
            dense=True,
            diag_shift=args.diag_shift,
            diag_scale=None,
            holomorphic=True,
            chunk_size=None,
        )
        from jax.sharding import AxisType

        # --- dense QGT: single -> replicated (_qgt_to_dense); distributed-eigh -> map-side
        # sharded build. The distributed solve asserts a pre-padded sharded Sd, so any
        # non-single mode MUST build with sharded_to_dense. ---
        mesh = None
        if args.mode == "single":
            Sd = jax.block_until_ready(_qgt_to_dense(S))
        else:
            mesh = jax.make_mesh((ndev,), ("S",), axis_types=(AxisType.Auto,))
            with jax.sharding.set_mesh(mesh):
                Sd = jax.block_until_ready(sharded_to_dense(S))
        print(f"[worker] Sd built: shape={Sd.shape} sharding={Sd.sharding}")

        # E_loc / importance_weights: their *values* don't affect solve timing (the O(n^3)
        # eigh dominates; F/reassembly are O(n^2)), so synthesize them at the right shape
        # instead of building the Hamiltonian.
        key = jax.random.PRNGKey(SEED)
        E_loc = (
            jax.random.normal(key, samples_q.shape[:-1])
            + 1j * jax.random.normal(jax.random.fold_in(key, 1), samples_q.shape[:-1])
        )
        iw = jnp.ones(samples_q.shape[:-1], dtype=float)
        rhs_coeff = -1.0j  # real-time TDVP factor; timing-irrelevant
        snr_atol = None
        def solve(distributed):
            return _solve_dense(
                vstate.parameters, vstate.n_samples, E_loc, S, Sd, iw,
                rhs_coeff, args.rcond, args.rcond_smooth, snr_atol, distributed,
            )

        if args.mode == "single":
            compile_time, times = bench(lambda: solve(False), args.warmup, args.reps)
        else:  # distributed or sharded -> distributed_eigh solve inside the mesh context
            def solve_dist():
                with jax.sharding.set_mesh(mesh):
                    return solve(True)

            compile_time, times = bench(solve_dist, args.warmup, args.reps)

        rec.update(
            t_min=min(times),
            t_med=statistics.median(times),
            t_mean=statistics.mean(times),
            t_std=statistics.pstdev(times),
            compile_time=compile_time,
        )
        print(f"[worker] mode={args.mode} n_params={n_params}: t_med={rec['t_med']:.4f}s")
    except Exception as e:  # OOM (RESOURCE_EXHAUSTED) or any build/solve failure
        msg = repr(e)
        is_oom = "RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()
        rec["oom"] = bool(is_oom)
        rec["error"] = msg[:500]
        print(f"[worker] mode={args.mode} n_params={n_params}: "
              f"{'OOM' if is_oom else 'FAILED'} -> {msg[:200]}")

    os.makedirs(out_dir, exist_ok=True)
    np.savez(out_path, **rec)
    print(f"[worker] wrote {out_path}")


def bench(fn, n_warmup, n_reps):
    """Time ``fn`` (zero-arg -> JAX pytree). Returns (compile_time, [steady times])."""
    import jax

    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    compile_time = time.perf_counter() - t0
    for _ in range(n_warmup):
        jax.block_until_ready(fn())
    times = []
    for _ in range(n_reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - t0)
    return compile_time, times


# ======================================================================================
# Parent: orchestrate one worker subprocess per (size, mode).
# ======================================================================================
def run_parent(args):
    sites = [int(s) for s in args.sites.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    orders = tuple(int(o) for o in args.orders.split(","))
    os.makedirs(args.out, exist_ok=True)

    # ndev_<k> subdir per mode (single=1; sharded=devices used). Only the sharded case
    # needs the real box device count; count it lazily (once) so a single-only sweep
    # never spins up JAX.
    visible = [None]

    def ndev_for(mode):
        if mode == "single":
            return 1
        if visible[0] is None:
            visible[0] = _count_visible_devices()
        return ndev_used(mode, args.n_devices, visible[0])

    print(f"orchestrating {len(sites)} size(s) x {len(modes)} mode(s) -> {args.out}")
    for mode in modes:
        out_dir = os.path.join(args.out, f"ndev_{ndev_for(mode)}")
        for N in sites:
            n_params = n_params_for(N, orders, args.rank, args.legacy_rank_k2)
            out_path = os.path.join(out_dir, f"solve_{mode}_np{n_params}.npz")
            if os.path.exists(out_path) and not args.force:
                print(f"  skip {mode} N={N} (n_params={n_params}); {out_path} exists")
                continue

            env = dict(os.environ)
            env["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
            # single-device: pin to one GPU; sharded: expose all (or --n-devices).
            if mode == "single":
                env["CUDA_VISIBLE_DEVICES"] = "0"
            elif args.n_devices and args.n_devices > 0:
                env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(args.n_devices))

            cmd = [
                sys.executable, os.path.abspath(__file__), "--worker",
                "--mode", mode, "--sites", str(N),
                "--orders", args.orders, "--rank", str(args.rank),
                *([] if args.legacy_rank_k2 else ["--no-legacy-rank-k2"]),
                "--n-samples", str(args.n_samples),
                "--rcond", str(args.rcond), "--rcond-smooth", str(args.rcond_smooth),
                "--diag-shift", str(args.diag_shift),
                "--n-devices", str(args.n_devices),
                "--reps", str(args.reps), "--warmup", str(args.warmup),
                "--check-max", str(args.check_max),
                "--out", args.out,
            ]
            if args.force:
                cmd.append("--force")
            print(f"\n>>> {mode} N={N} n_params={n_params}")
            subprocess.run(cmd, env=env, check=False)
    print("\ndone. results under", args.out)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sites", default=",".join(map(str, DEFAULT_SITES)),
                   help="comma list of spin counts N to sweep")
    p.add_argument("--orders", default=",".join(map(str, DEFAULT_ORDERS)),
                   help="comma list of Jastrow orders (all >=2 assumed for the size formula)")
    p.add_argument("--rank", type=int, default=DEFAULT_RANK)
    p.add_argument("--no-legacy-rank-k2", dest="legacy_rank_k2", action="store_false",
                   help="drop the redundant rank axis on the k=2 Jastrow block")
    p.add_argument("--modes", default="single,sharded",
                   help="comma list of {single, sharded}; 'sharded' is the distributed-eigh "
                        "path with the QGT built row-sharded map-side (sharded_to_dense)")
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES,
                   help="QGT-build sample count (not timed)")
    p.add_argument("--n-devices", type=int, default=0,
                   help="devices for the distributed path (0 = all visible)")
    p.add_argument("--rcond", type=float, default=DEFAULT_RCOND)
    p.add_argument("--rcond-smooth", type=float, default=DEFAULT_RCOND_SMOOTH)
    p.add_argument("--diag-shift", type=float, default=DEFAULT_DIAG_SHIFT)
    p.add_argument("--reps", type=int, default=N_REPS)
    p.add_argument("--warmup", type=int, default=N_WARMUP)
    p.add_argument("--check-max", type=int, default=8000,
                   help="max n_params at which to run the distributed-vs-single dw guard")
    p.add_argument("--out", default="data/eigh")
    p.add_argument("--force", action="store_true", help="overwrite existing npz")
    # internal: worker mode runs exactly one (size, mode)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--mode", default=None, help=argparse.SUPPRESS)
    return p


def main():
    args = build_parser().parse_args()
    if args.worker:
        # In worker mode --sites is a single N.
        args.sites = int(args.sites)
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
