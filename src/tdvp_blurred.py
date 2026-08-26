# Copyright 2020, 2021  The NetKet Authors - All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Callable

from functools import partial

import os
import time

import jax
import jax.numpy as jnp
import numpy as np

from netket import stats
from netket.operator import AbstractOperator
from netket.optimizer.qgt.qgt_jacobian import QGTJacobian_DefaultConstructor
from netket.optimizer.qgt.qgt_jacobian_dense import convert_tree_to_dense_format
from netket.utils.api_utils import partial_from_kwargs
from netket.vqs import VariationalState, VariationalMixedState, MCState
from netket.jax import tree_cast
from netket.utils import HashablePartial
from netket.jax._utils_tree import tree_conj, tree_dot
from collections.abc import Sequence

from netket.experimental.driver.tdvp_common import TDVPBaseDriver, odefun
from netket.experimental.dynamics._solver import AbstractSolver

from .tdvp_utils import (
    make_monitor_dict,
    blurred_sample,
    reweight_on_configs,
    ess_from_weights,
)
from jax.sharding import PartitionSpec as P

import platform

system = platform.system()
if system == "Linux":
    try:
        from jaxmg import syevd

        JAXMG_ENABLED = True
    except ModuleNotFoundError:
        JAXMG_ENABLED = False
else:
    JAXMG_ENABLED = False

# How many times to retry a step whose ODE-integrator attempt errored
_MAX_STEP_RETRIES = 1


class TDVPBlurred(TDVPBaseDriver):
    r"""
    Variational time evolution based on the time-dependent variational principle which,
    when used with Monte Carlo sampling via :class:`netket.vqs.MCState`, is the time-dependent VMC
    (t-VMC) method.

    This driver, which only works with standard MCState variational states, uses the regularization
    procedure described in Medvidovic et. al.'s https://arxiv.org/abs/2212.11289.

    With the force vector

    .. math::

        F_k=\langle \mathcal O_{\theta_k}^* E_{loc}^{\theta}\rangle_c

    and the quantum Fisher matrix

    .. math::

        S_{k,k'} = \langle \mathcal O_{\theta_k} (\mathcal O_{\theta_{k'}})^*\rangle_c

    and for real parameters :math:`\theta\in\mathbb R`, the TDVP equation reads

    .. math::

        q\big[S_{k,k'}\big]\theta_{k'} = -q\big[xF_k\big]

    Here, either :math:`q=\text{Re}` or :math:`q=\text{Im}` and :math:`x=1` for ground state
    search or :math:`x=i` (the imaginary unit) for real time dynamics.

    For ground state search a regularization controlled by a parameter :math:`\rho` can be included
    by increasing the diagonal entries and solving

    .. math::

        q\big[(1+\rho\delta_{k,k'})S_{k,k'}\big]\theta_{k'} = -q\big[F_k\big]

    The `TDVP` class solves the TDVP equation by computing a pseudo-inverse of :math:`S` via
    eigendecomposition yielding

    .. math::

        S = V\Sigma V^\dagger

    with a diagonal matrix :math:`\Sigma_{kk}=\sigma_k`
    Assuming that :math:`\sigma_1` is the smallest eigenvalue, the pseudo-inverse is constructed
    from the regularized inverted eigenvalues

    .. math::

        \tilde\sigma_k^{-1}=\frac{1}{\Big(1+\big(\frac{\epsilon_{SVD}}{\sigma_j/\sigma_1}\big)^6\Big)\Big(1+\big(\frac{\epsilon_{SNR}}{\text{SNR}(\rho_k)}\big)^6\Big)}

    with :math:`\text{SNR}(\rho_k)` the signal-to-noise ratio of
    :math:`\rho_k=V_{k,k'}^{\dagger}F_{k'}` (see
    `[arXiv:1912.08828] <https://arxiv.org/pdf/1912.08828.pdf>`_ for details).


    .. note::

        This TDVP Driver uses the time-integrators from the `nkx.dynamics` module, which are
        automatically executed under a `jax.jit` context.

        When running computations on GPU, this can lead to infinite hangs or extremely long
        compilation times. In those cases, you might try setting the configuration variable
        `nk.config.netket_experimental_disable_ode_jit = True` to mitigate those issues.

    """

    def __init__(
        self,
        operator: AbstractOperator,
        variational_state: VariationalState,
        integrator: AbstractSolver = None,
        *,
        q: float = 0.1,
        t0: float = 0.0,
        propagation_type: str = "real",
        holomorphic: bool | None = None,
        diag_shift: float | None = 0.0,
        diag_scale: float | None = None,
        error_norm: str | Callable = "qgt",
        rcond: float = 1e-14,
        rcond_smooth: float = 1e-8,
        snr_atol: float | None = None,
        sampling_state: VariationalState = None,
        distributed_eigh: bool = False,
        log_eigenvalues: bool = False,
        cache_within_step: bool = True,
        error_dump_dir: str | None = None,
    ):
        r"""
        Initializes the time evolution driver.

        Args:
            operator: The generator of the dynamics (Hamiltonian for pure states,
                Lindbladian for density operators).
            variational_state: The variational state.
            integrator: Configuration of the algorithm used for solving the ODE.
            t0: Initial time at the start of the time evolution.
            propagation_type: Determines the equation of motion: "real"  for the
                real-time Schrödinger equation (SE), "imag" for the imaginary-time SE.
            error_norm: Norm function used to calculate the error with adaptive integrators.
                Can be either "euclidean" for the standard L2 vector norm :math:`w^\dagger w`,
                "maximum" for the maximum norm :math:`\max_i |w_i|`
                or "qgt", in which case the scalar product induced by the QGT :math:`S` is used
                to compute the norm :math:`\Vert w \Vert^2_S = w^\dagger S w` as suggested
                in PRL 125, 100503 (2020).
                Additionally, it possible to pass a custom function with signature
                :code:`norm(x: PyTree) -> float`
                which maps a PyTree of parameters :code:`x` to the corresponding norm.
                Note that norm is used in jax.jit-compiled code.
            holomorphic: a flag to indicate that the wavefunction is holomorphic.
            diag_shift: diagonal shift of the quantum geometric tensor (QGT)
            diag_scale: If not None rescales the diagonal shift of the QGT
            rcond : Cut-off ratio for small singular :math:`\sigma_k` values of the
                Quantum Geometric Tensor. For the purposes of rank determination,
                singular values are treated as zero if they are smaller than rcond times
                the largest singular value :code:`\sigma_{max}`.
            rcond_smooth : Smooth cut-off ratio for singular values of the Quantum Geometric
                Tensor. This regularization parameter used with a similar effect to `rcond`
                but with a softer curve. See :math:`\epsilon_{SVD}` in the formula
                above.
            snr_atol: Noise regularisation absolute tolerance (Schmitt
                PRL 125, 100503, :math:`\epsilon_{SNR}`). Eigen-directions of the QGT
                whose force projection has a signal-to-noise ratio below this value are
                (soft) truncated via ``1/(1+(snr_atol/snr)^6)``. ``None`` disables the
                SNR filter and reproduces the plain smooth-pseudoinverse solve.
                Applied on both the single-device and distributed-eigh solve paths.
            error_dump_dir: Directory into which the latest TDVP diagnostics (``self._info``,
                plus time/step/dt/energy context) are written as JSON if the ODE integrator
                errors out (e.g. an invalid/NaN ``dt``), to help diagnose the failure. Defaults
                to the current working directory.

        """
        self.propagation_type = propagation_type
        if isinstance(variational_state, VariationalMixedState):
            # assuming Lindblad Dynamics
            # TODO: support density-matrix imaginary time evolution
            if propagation_type == "real":
                self._loss_grad_factor = 1.0
            else:
                raise ValueError(
                    "only real-time Lindblad evolution is supported for " "mixed states"
                )
        else:
            if propagation_type == "real":
                self._loss_grad_factor = -1.0j
            elif propagation_type == "imag":
                self._loss_grad_factor = -1.0
            else:
                raise ValueError("propagation_type must be one of 'real', 'imag'")

        self.rcond = rcond
        self.rcond_smooth = rcond_smooth
        self.snr_atol = snr_atol

        self.diag_shift = diag_shift
        self.holomorphic = holomorphic
        self.diag_scale = diag_scale

        self.log_eigenvalues = log_eigenvalues

        # Where to dump diagnostics if the ODE integrator errors (see `_dump_error_info`).
        self.error_dump_dir = error_dump_dir

        # When True, draw one MCMC sample (+ blur, if q>0) at the first RK stage of each
        # ODE step and reweight it across the remaining stages instead of resampling.
        self.cache_within_step = cache_within_step
        self._cached_samples = None
        self._cached_blur_w = None
        self._cached_logpsi_ref = None

        # Armed by `_iter` after an errored integrator attempt: makes `odefun_custom`
        # dump the raw per-sample `pdf`/`E_loc` of the retry attempt
        self._dump_samples = False
        self._dump_attempt = 0

        # Per-ODE-step wall-clock timing 
        self._step_time_total = None
        self._step_times = {"mcmc": 0.0, "blur": 0.0, "qgt": 0.0, "solve": 0.0}

        self._monitor = {}
        self._info = {}
        # Per-RK-stage snapshots of `_info` within the current ODE step.
        self._info_stages = {}
        if not (0 <= q < 1):
            raise ValueError(f"`q` must satisfy 0 < q < 1, received {q}")
        self.q = q
        if distributed_eigh and not JAXMG_ENABLED:
            raise ImportError(
                "distributed_eigh=True requires jaxmg to be installed and enabled. "
                "Please install jaxmg (pip install jaxmg) and set the environment variable "
                "ENABLE_JAXMG=1 before running. This feature is only available on Linux systems."
            )
        self.distributed_eigh = distributed_eigh

        if sampling_state is not None:
            if not isinstance(sampling_state, VariationalState):
                raise ValueError(
                    f"Expected `sampling_state` to be a VariationalState, received {type(sampling_state)}"
                )
            # Check that sampling_state and variational_state have the same parameter structure
            sampling_structure = jax.tree_util.tree_structure(sampling_state.parameters)
            variational_structure = jax.tree_util.tree_structure(
                variational_state.parameters
            )
            if sampling_structure != variational_structure:
                raise ValueError(
                    f"Parameter structures of sampling_state and variational_state do not match. "
                    f"sampling_state structure: {sampling_structure}, "
                    f"variational_state structure: {variational_structure}"
                )
            # Store the dtype of sampling_state parameters
            sampling_leaves = jax.tree_util.tree_leaves(sampling_state.parameters)
            self.sampling_dtype = sampling_leaves[0].dtype
            self.sampling_state = sampling_state
        else:
            self.sampling_state = None
            self.sampling_dtype = None

        super().__init__(
            operator, variational_state, integrator, t0=t0, error_norm=error_norm
        )

    def _iter(
        self,
        T: float,
        tstops: Sequence[float] | None = None,
        callback: Callable | None = None,
    ):
        """
        Implementation of :code:`iter`. This method accepts and additional `callback` object, which
        is called after every accepted step.
        """
        t_end = self.t + T
        if tstops is not None and (
            np.any(np.less(tstops, self.t)) or np.any(np.greater(tstops, t_end))
        ):
            raise ValueError(
                f"All tstops must be in range [t, t + T]=[{self.t}, {t_end}]"
            )

        if tstops is not None and len(tstops) > 0:
            tstops = np.sort(tstops)
            always_stop = False
        else:
            tstops = []
            always_stop = True

        while self.t < t_end:
            if always_stop or (
                len(tstops) > 0
                and (np.isclose(self.t, tstops[0]) or self.t > tstops[0])
            ):
                self._stop_count += 1
                yield self.t
                tstops = tstops[1:]

            # Reset the per-step section accumulator before the step
            self._step_times = {"mcmc": 0.0, "blur": 0.0, "qgt": 0.0, "solve": 0.0}
            _t0 = time.perf_counter()
            step_accepted = False
            # Consecutive errored attempts at the current step.
            n_failed = 0
            while not step_accepted:
                if not always_stop and len(tstops) > 0:
                    max_dt = tstops[0] - self.t
                else:
                    max_dt = None
                dt_tried = self._integrator._state.dt
                step_accepted = self._integrator.step(max_dt=max_dt)
                if self._integrator.errors:
                    message = self._integrator.errors.message()
                    n_failed += 1
                    if n_failed > _MAX_STEP_RETRIES:
                        # Don't leave the driver armed if the exception is caught.
                        self._dump_samples = False
                        self._dump_error_info(message)
                        raise RuntimeError(
                            f"ODE integrator: {message} "
                            f"(failed {n_failed} consecutive attempts)"
                        )
                    self._integrator._state = self._integrator._state.replace(
                        dt=dt_tried
                    )
                    # Reset to pre-failed state.
                    w_ok = self._integrator._state.y
                    self.state.parameters = w_ok
                    if self.sampling_state is not None:
                        self.sampling_state.parameters = tree_cast(
                            w_ok, self.sampling_state.parameters
                        )
                    # Move the chains forward a bit.
                    _s = None
                    for _i in range(5):
                        if self.sampling_state is not None:
                            self.sampling_state.reset()
                            _s = self.sampling_state.samples
                        else:
                            self.state.reset()
                            _s = self.state.samples
                    # The rounds are dispatched lazily; make them finish before retrying.
                    jax.block_until_ready(_s)
                    # Arm the per-sample dump for the retry attempt: `odefun_custom`
                    # writes its `pdf`/`E_loc` for every RK stage.
                    self._dump_samples = True
                    self._dump_attempt = n_failed
                    print(
                        f"[tdvp] ODE integrator error at t={self.t} "
                        f"(retry {n_failed}/{_MAX_STEP_RETRIES}): {message}; "
                        f"retrying with dt={dt_tried} and dumping per-stage samples"
                    )
                    step_accepted = False
                else:
                    n_failed = 0
                    self._dump_samples = False
            # Ensure the step's async work is finished before stopping the clock.
            jax.block_until_ready(self.state.parameters)
            self._step_time_total = time.perf_counter() - _t0
            self._step_count += 1
            # optionally call callback
            if callback:
                callback()

        # Yield one last time if the remaining tstop is at t_end
        if (always_stop and np.isclose(self.t, t_end)) or (
            len(tstops) > 0 and np.isclose(tstops[0], t_end)
        ):
            yield self.t

    def _dump_error_info(self, message):
        """Dump the TDVP diagnostics to JSON when the ODE integrator errors"""
        import json

        def _to_jsonable(v):
            a = np.asarray(jax.device_get(v))
            return a.item() if a.ndim == 0 else a.ravel().tolist()

        dump = {"error": str(message), "t": float(self.t), "step": int(self._step_count)}

        dt = getattr(self.integrator._state, "dt", None)
        if dt is not None:
            dump["dt"] = _to_jsonable(dt)

        loss = getattr(self, "_loss_stats", None)
        if loss is not None:
            dump["energy_real"] = float(loss.mean.real)
            dump["energy_imag"] = float(loss.mean.imag)
            dump["variance"] = float(loss.variance)

        # One entry per RK stage of the failing step, in stage order. 
        stages = self._info_stages or ({0: self._info} if self._info else {})
        dump["stages"] = [
            {"stage": k, **{kk: _to_jsonable(vv) for kk, vv in stages[k].items()}}
            for k in sorted(stages)
        ]

        out_dir = self.error_dump_dir or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"tdvp_error_step{self._step_count}.json")
        with open(path, "w") as f:
            json.dump(dump, f, indent=2)
        print(f"[tdvp] ODE integrator error; dumped diagnostics to {path}")
        return path

    def _dump_stage_samples(self, stage, t, pdf, E_loc, ess, w_mean):
        """Dump the raw per-sample ``pdf`` and ``E_loc`` of one RK stage.

        Called from :func:`odefun_custom` only while ``self._dump_samples`` is armed,
        i.e. on an attempt that directly follows an errored integrator step, to expose
        the spurious samples that blew up the force/QGT. Arrays keep their
        ``(n_chains, n_samples_per_chain)`` shape so a single stuck chain is visible.
        Written as ``npz`` (not JSON like `_dump_error_info`) because these are the
        full sample arrays. Never raises: a failed dump must not mask the real error.
        """
        try:
            out_dir = self.error_dump_dir or os.getcwd()
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(
                out_dir,
                f"tdvp_samples_step{self._step_count}"
                f"_att{self._dump_attempt}_stage{stage}.npz",
            )
            _get = lambda x: np.asarray(jax.device_get(x))  # noqa: E731
            np.savez_compressed(
                path,
                pdf=_get(pdf),
                E_loc=_get(E_loc),
                t=_get(t),
                ess=_get(ess),
                w_mean=_get(w_mean),
                stage=np.asarray(stage),
                step=np.asarray(self._step_count),
                attempt=np.asarray(self._dump_attempt),
            )
            print(f"[tdvp] dumped stage-{stage} samples to {path}")
            return path
        except Exception as e:  # never mask the integrator error with a dump failure
            print(f"[tdvp] WARNING: failed to dump stage-{stage} samples: {e!r}")
            return None

    def _log_additional_data(self, log_dict, step):
        super()._log_additional_data(log_dict, step)
        # Log every RK stage's diagnostics under stage-prefixed keys
        stages = self._info_stages or ({0: self._info} if self._info else {})
        for stage, info in stages.items():
            for k, v in info.items():
                if not self.log_eigenvalues and k in ["snr", "snr_F", "ev", "ev_reg"]:
                    continue
                log_dict[f"{stage}/{k}"] = v
        if hasattr(self.integrator._state, "dt"):
            log_dict["dt"] = self.integrator._state.dt
        log_dict["step"] = self._step_count
        log_dict["q"] = self.q
        # Live sampler sweep size (fixed for the whole run).
        sampler = getattr(self.state, "sampler", None)
        if sampler is not None and hasattr(sampler, "sweep_size"):
            log_dict["sweep_size"] = int(sampler.sweep_size)
        # Parallel-tempering health. 
        ss = getattr(self.state, "sampler_state", None)
        if ss is not None and hasattr(ss, "normalized_diffusion"):
            # normalized_diffusion divides by exchange_steps → NaN before any swap step.
            if int(ss.exchange_steps) > 0:
                log_dict["pt_norm_diffusion"] = float(ss.normalized_diffusion)
            log_dict["pt_norm_position"] = float(ss.normalized_position)
            acc = getattr(ss, "acceptance", None)  # None before any sampling step
            if acc is not None:
                log_dict["pt_local_accept"] = float(acc)
            # Per-temperature *local* acceptance
            nab = getattr(ss, "n_accepted_per_beta", None)
            nsteps = int(getattr(ss, "n_steps", 0))
            if nab is not None and nsteps > 0:
                try:
                    nab = np.asarray(nab, float)          # (n_chains, n_replicas)
                    beta = np.asarray(ss.beta, float)
                    order = np.argsort(beta, axis=1)       # ascending: hot -> cold
                    nab_s = np.take_along_axis(nab, order, axis=1)
                    rate = nab_s.sum(axis=0) / nsteps      # per-rung, aggregated over chains
                    log_dict["pt_accept_hot"] = float(rate[0])       # smallest beta
                    log_dict["pt_accept_cold"] = float(rate[-1])     # beta=1 (== pt_local_accept)
                    log_dict["pt_accept_ladder_min"] = float(rate.min())
                    log_dict["pt_accept_ladder_med"] = float(np.median(rate))
                    log_dict["pt_accept_per_beta"] = rate            # full profile, hot->cold
                except Exception:
                    pass
        solver_state = getattr(self.integrator._state, "solver_state", None)
        if solver_state is not None:
            if hasattr(solver_state, "n_iter"):
                log_dict["n_iter"] = solver_state.n_iter
            if hasattr(solver_state, "rel_err"):
                log_dict["rel_err"] = solver_state.rel_err
        # Per-step wall-clock timing (None on the initial pre-step log point).
        if self._step_time_total is not None:
            log_dict["time/total"] = self._step_time_total
            for k, v in self._step_times.items():
                log_dict[f"time/{k}"] = v


def _distributed_eigh_padding_and_tile(n, num_devices):
    """Compute the padding and cuSolverMg tile size for the distributed eigh.

    cuSolverMg row-shards the matrix across ``num_devices`` and further splits
    each shard into tiles of width ``T_A``. This returns ``(n_pad, T_A)`` such
    that, for a matrix of size ``n``:

      * ``n + n_pad`` is divisible by ``num_devices`` (valid row sharding), and
      * the per-device shard ``(n + n_pad) // num_devices`` is an exact multiple
        of ``T_A`` with ``tile_min <= T_A <= tile_max``, so syevd does not have
        to re-tile and copy the matrix internally.

    We first look for the largest tile in ``[tile_min, tile_max]`` that divides
    the minimally-padded shard exactly (no extra padding). If none exists we
    fall back to ``T_A = tile_min`` and pad each shard up to the next multiple
    of ``tile_min``, which adds the least padding while keeping the tiling exact.
    """
    # Minimal padding so the rows can be evenly sharded across the devices.
    n_pad = (-n) % num_devices
    shard = (n + n_pad) // num_devices

    # Largest exact divisor in range needs no extra padding.
    for T_A in range(min(1024, max(shard, 256)), 256 - 1, -1):
        if shard % T_A == 0:
            return n_pad, T_A

    # No exact divisor (or the shard is smaller than tile_min): pad each shard
    # up to the next multiple of tile_min so the tiling divides evenly.
    T_A = 256
    padded_shard = -(-shard // T_A) * T_A
    n_pad = padded_shard * num_devices - n
    return n_pad, T_A

@jax.jit
def _qgt_to_dense(S):
    """Materialize the dense QGT matrix"""
    Sd = S.to_dense()
    return Sd


@jax.jit
def sharded_to_dense(S):
    """Row-sharded, map-side dense QGT for the ``distributed_eigh`` path."""
    mesh = jax.sharding.get_abstract_mesh()
    num_devices = mesh.shape["S"]
    n = S.O.shape[-1]
    n_pad, _ = _distributed_eigh_padding_and_tile(n, num_devices)
    row = P("S", None)

    if S.scale is None:
        O = S.O
        dvals = S.diag_shift * jnp.ones(n)
    else:
        O = S.O * S.scale[jnp.newaxis, :]
        dvals = S.diag_shift * S.scale**2

    def pad_cols(A):  # append n_pad zero columns on the params (last) axis
        return A if n_pad == 0 else jnp.pad(A, ((0, 0), (0, n_pad)))

    if S.mode == "imag":
        flip_sign = jnp.array([1, -1]).reshape(1, 2, 1)
        Ol = pad_cols((flip_sign * O).reshape(-1, n))
        Or = pad_cols(jnp.flip(O, axis=1).reshape(-1, n))
        G = (jax.lax.with_sharding_constraint(Ol, P(None, "S")).T
             @ jax.lax.with_sharding_constraint(Or, P()))
    else:
        O = pad_cols(O.reshape(-1, n))
        G = (jax.lax.with_sharding_constraint(O, P(None, "S")).conj().T
             @ jax.lax.with_sharding_constraint(O, P()))
    G = jax.lax.with_sharding_constraint(G, row)
    idx = jnp.arange(n)  # diag_shift on the TRUE diagonal only; padded diagonal stays 0
    G = G.at[idx, idx].add(dvals.astype(G.dtype))
    return jax.lax.with_sharding_constraint(G, row)


@partial(
    jax.jit,
    static_argnames=(
        "n_samples",
        "rcond",
        "rcond_smooth",
        "snr_atol",
        "distributed_eigh",
    ),
)
def _solve_dense(
    parameters,
    n_samples,
    E_loc,
    S,
    Sd,
    importance_weights,
    rhs_coeff,
    rcond,
    rcond_smooth,
    snr_atol,
    distributed_eigh,
):
    """Eigendecomposition + regularized TDVP solve on the pre-materialized dense
    QGT ``Sd`` (from ``_qgt_to_dense``).

    Solve ``S dtheta = rhs_coeff * F`` with a hard ``rcond`` cutoff + the
    ``rcond_smooth`` soft eigenvalue filter, optionally multiplied by the Schmitt
    PRL 125.100503 per-eigendirection force-SNR soft cutoff
    ``1/(1+(snr_atol/snr)^6)``.

    - ``distributed_eigh=False``: full :func:`jax.numpy.linalg.eigh` of ``Sd`` and
      the eigenbasis solve, including the ``snr_atol`` filter. With ``snr_atol=None``
      this is numerically the smooth pseudo-inverse
      (:func:`netket.optimizer.solver.pinv_smooth`).
    - ``distributed_eigh=True``: multi-GPU ``syevd`` + the same eigenbasis solve and
      ``snr_atol`` filter applied by hand on the padded spectrum (the padding /
      zero-eigenvalue directions are dropped from the logged ``ev``/``snr``/``ev_reg``).

    Compute epsilon squared and force SNR.
    """
    w_mean = jnp.mean(importance_weights)
    pdf = importance_weights / w_mean
    E = stats.statistics(pdf * E_loc)
    # replace variance with importance weighed one
    weighted_variance = jnp.mean(pdf * jnp.abs(E_loc - E.mean) ** 2)
    E = E.replace(variance=weighted_variance)
    ΔE_loc = E_loc.reshape(-1, 1) - E.mean

    stack_jacobian = S.mode == "complex"

    O = S.O
    if stack_jacobian:
        O = O.reshape(-1, 2, S.O.shape[-1])
        O = O[:, 0, :] + 1j * O[:, 1, :]
    pdf = jnp.reshape(pdf, (-1, 1))
    O = O * jnp.sqrt(
        pdf / pdf.size
    )  # O is already multiplied with sqrt(pdf) so now O * sqrt(pdf)->O * pdf

    OEdata = O.conj() * ΔE_loc
    # SNR of the force estimator F = sum_i O_i^* ΔE_i (parameter basis, kept for
    # monitoring; needs no eigendecomposition).
    OE_mean = jnp.mean(OEdata, axis=0)
    OE_var = jnp.var(OEdata, axis=0)
    eps = jnp.finfo(O.dtype).eps
    snr_F = jnp.where(
        OE_var <= eps,
        jnp.inf,
        jnp.abs(OE_mean) * jnp.sqrt(n_samples) / jnp.sqrt(OE_var + eps),
    )
    F = jnp.sum(OEdata, axis=0)
    # True parameter count from the force vector
    n_params = F.shape[0]

    if distributed_eigh:
        # Multi-GPU eigendecomposition (jaxmg syevd); apply the Schmitt filter by
        # hand since pinv_smooth would run its own (non-distributed) eigh. Same
        # eigenbasis force-SNR filter as the else-branch, on the padded spectrum.
        mesh = jax.sharding.get_abstract_mesh()
        num_devices = mesh.shape["S"]
        n_pad, T_A = _distributed_eigh_padding_and_tile(n_params, num_devices)
        M = n_params + n_pad
        b = rhs_coeff * F
        if n_pad:
            b = jnp.pad(b, (0, n_pad))
            # pad the Jacobian columns to M so the eigenbasis projection Q = V^H O^T
            # is conformable with the M x M eigenvectors from syevd.
            O = jnp.pad(O, ((0, 0), (0, n_pad)))

        Sd = jax.lax.with_sharding_constraint(Sd, P("S", None))
        ev, V = syevd(Sd, T_A=T_A, mesh=mesh, in_specs=(P("S", None),))
        ev = jax.lax.with_sharding_constraint(ev, P())
        V = jax.lax.with_sharding_constraint(V, P(None, "S"))
        # eigenbasis force-SNR (Eq. 21). rho already carries rhs_coeff via b; since
        # |rhs_coeff|=1 this matches the else-branch's |V^H F| in the snr magnitude.
        rho = V.conj().T @ b
        Q = jnp.tensordot(V.conj().T, O.T, axes=1).T
        QEdata = Q.conj() * ΔE_loc
        sigma_k = jnp.maximum(jnp.sqrt(jnp.var(QEdata, axis=0)), rcond)
        snr = jnp.where(
            sigma_k <= eps,
            jnp.inf,
            jnp.abs(rho) * jnp.sqrt(n_samples) / sigma_k,
        )
        ev_inv = jnp.where(jnp.abs(ev / ev[-1]) > rcond, jnp.reciprocal(ev), 0.0)
        regularizer = 1.0 / (1.0 + (rcond_smooth / jnp.abs(ev / ev[-1])) ** 6)
        if snr_atol is not None:
            regularizer = regularizer * (1.0 / (1.0 + (snr_atol / snr) ** 6))
        update = (V @ (ev_inv * regularizer * rho))[:n_params]
        ev_reg = jnp.where(
            ev_inv * regularizer < 1.0 / rcond,
            1.0 / (ev_inv * regularizer),
            jnp.nan,
        )
        # drop the (zero-eigenvalue) padding directions from the logged spectra;
        # syevd sorts ascending so they are the first n_pad entries.
        ev = ev[n_pad:]
        snr = snr[n_pad:]
        ev_reg = ev_reg[n_pad:]
    else:
        # Restored Schmitt PRL 125.100503 eigenbasis solve (was replaced by
        # pinv_smooth in cf7f3dd): hard rcond cutoff + rcond_smooth soft filter,
        # optionally times the per-direction force-SNR soft cutoff
        # 1/(1+(snr_atol/snr)^6). snr_atol=None => identical to the pinv_smooth solve.
        ev, V = jnp.linalg.eigh(Sd)
        rho = V.conj().T @ F                          # force in the S eigenbasis
        Q = jnp.tensordot(V.conj().T, O.T, axes=1).T  # Jacobian in the S eigenbasis
        QEdata = Q.conj() * ΔE_loc
        # per-eigendirection SNR (Eq. 21); guard sigma_k -> 0 (netket#1959/#1960)
        sigma_k = jnp.maximum(jnp.sqrt(jnp.var(QEdata, axis=0)), rcond)
        snr = jnp.where(
            sigma_k <= eps,
            jnp.inf,
            jnp.abs(rho) * jnp.sqrt(n_samples) / sigma_k,
        )
        ev_inv = jnp.where(jnp.abs(ev / ev[-1]) > rcond, 1.0 / ev, 0.0)
        regularizer = 1.0 / (1.0 + (rcond_smooth / jnp.abs(ev / ev[-1])) ** 6)
        if snr_atol is not None:
            regularizer = regularizer * (1.0 / (1.0 + (snr_atol / snr) ** 6))
        update = (V @ (ev_inv * regularizer * rhs_coeff * rho))[:n_params]
        ev_reg = jnp.where(
            ev_inv * regularizer < 1.0 / rcond,
            1.0 / (ev_inv * regularizer),
            jnp.nan,
        )

    y, reassemble = convert_tree_to_dense_format(parameters, S.mode)
    complex_mode = jnp.iscomplexobj(y)
    # TDVP ERROR #
    update_tree = reassemble(update if complex_mode else update.real)
    force_tree = reassemble(F if complex_mode else F.real)
    update_tree_conj = tree_conj(update_tree)
    rmd_1 = tree_dot(update_tree_conj, S @ update_tree)
    rmd_2 = tree_dot(
        update_tree_conj, jax.tree.map(lambda x: rhs_coeff * x, force_tree)
    )
    epsilon_squared = 1 + (rmd_1.real - 2 * rmd_2.real) / (E.variance + 1e-30)
    # If parameters are real, then take only real part of the gradient (if it's complex)
    dw = tree_cast(update_tree, parameters)

    if distributed_eigh:
        dw = jax.lax.with_sharding_constraint(dw, jax.tree.map(lambda _: P(), dw))
        snr = jax.lax.with_sharding_constraint(snr, P())
        snr_F = jax.lax.with_sharding_constraint(snr_F, P())
        ev = jax.lax.with_sharding_constraint(ev, P())
        ev_reg = jax.lax.with_sharding_constraint(ev_reg, P())

    return E, dw, epsilon_squared, snr, snr_F, ev, ev_reg


@odefun.dispatch
def odefun_custom(
    state: MCState, self: TDVPBlurred, t, w, *, stage=0
):  # noqa: F811
    # pylint: disable=protected-access

    state.parameters = w
    chunk_size = getattr(state, "chunk_size", None)

    # Generator and schedules
    op_t = self.generator(t)

    # --- timed sections: MCMC sampling and blur / local energies ---
    if self.q > 0:
        # The integrator calls this function once per Runge-Kutta tableau stage.
        #  Instead of sampling multiple times we reweight them to the current
        # w (no MCMC, no re-blur): weight_w(x) = w_blurred · |ψ_w(x) / ψ_{w0}(x)|².
        resample = (not self.cache_within_step) or stage == 0 or self._cached_samples is None
        if resample:
            state._sampler_seed, key = jax.random.split(state._sampler_seed, 2)

            # Full MCMC sample (timed on its own).
            _ts = time.perf_counter()
            if self.sampling_state is not None:
                self.sampling_state.parameters = tree_cast(
                    w, self.sampling_state.parameters
                )
                samples = self.sampling_state.samples
            else:
                samples = state.samples
            samples = jax.block_until_ready(samples)
            self._step_times["mcmc"] += time.perf_counter() - _ts

            # Blurred kernel + local energies.
            _ts = time.perf_counter()
            samples_q, importance_weights, E_loc, logpsi_ref = HashablePartial(
                blurred_sample,
                apply_fn=state._apply_fun,
                op=op_t,
                q=self.q,
                chunk_size=chunk_size,
            )(samples, key, w)
            if self.cache_within_step:
                self._cached_samples = samples_q
                self._cached_blur_w = importance_weights
                self._cached_logpsi_ref = logpsi_ref
        else:
            # Reuse the cached stage-0 configs, reweighting to the current
            # parameters w. No MCMC on this stage, so only `blur` accrues.
            _ts = time.perf_counter()
            samples_q = self._cached_samples
            E_loc, logpsi_now = reweight_on_configs(
                samples_q, w, state._apply_fun, op_t, chunk_size
            )
            ratio = jnp.exp(
                2.0 * (logpsi_now.real - self._cached_logpsi_ref.real)
            )
            importance_weights = self._cached_blur_w * ratio
    else:
        # Standard (q=0) path. With cache_within_step we draw one raw MCMC sample
        resample = (
            (not self.cache_within_step) or stage == 0 or self._cached_samples is None
        )
        if resample:
            # Full MCMC sample (timed on its own).
            _ts = time.perf_counter()
            if self.sampling_state is not None:
                self.sampling_state.parameters = tree_cast(
                    w, self.sampling_state.parameters
                )
                samples_q = self.sampling_state.samples
                state.sampler_state = state.sampler_state.replace(σ=samples_q)
            else:
                samples_q = state.samples
            samples_q = jax.block_until_ready(samples_q)
            self._step_times["mcmc"] += time.perf_counter() - _ts

            # Local energies + reference log-amplitude (for later reweighting).
            _ts = time.perf_counter()
            E_loc, logpsi_ref = reweight_on_configs(
                samples_q, w, state._apply_fun, op_t, chunk_size
            )
            importance_weights = jnp.ones(E_loc.size, dtype=float)
            if self.cache_within_step:
                self._cached_samples = samples_q
                self._cached_blur_w = importance_weights
                self._cached_logpsi_ref = logpsi_ref
        else:
            # Reuse the cached stage-0 configs, reweighting to the current
            # parameters w. No MCMC on this stage, so only `blur` accrues.
            _ts = time.perf_counter()
            samples_q = self._cached_samples
            E_loc, logpsi_now = reweight_on_configs(
                samples_q, w, state._apply_fun, op_t, chunk_size
            )
            ratio = jnp.exp(2.0 * (logpsi_now.real - self._cached_logpsi_ref.real))
            importance_weights = self._cached_blur_w * ratio
    samples_q, importance_weights, E_loc = jax.block_until_ready(
        (samples_q, importance_weights, E_loc)
    )
    self._step_times["blur"] += time.perf_counter() - _ts

    # Monitor ESS of the combined weights
    ess = ess_from_weights(importance_weights)
    # Normalize weights for use as a pdf
    w_mean = jnp.mean(importance_weights)
    pdf = importance_weights / w_mean

    # Get S-matrix
    pdf = pdf.reshape(samples_q.shape[:-1])
    E_loc = E_loc.reshape(samples_q.shape[:-1])
    # Retry attempt after an errored step: record the raw samples that produced it.
    if getattr(self, "_dump_samples", False):
        self._dump_stage_samples(stage, t, pdf, E_loc, ess, w_mean)
    # --- timed section: QGT construction + dense materialization. 
    _ts = time.perf_counter()
    self._S = partial_from_kwargs(
        QGTJacobian_DefaultConstructor,
        exclusive_arg_names=(("mode", "holomorphic")),
    )(
        state._apply_fun,
        state.parameters,
        state.model_state,
        samples_q,
        pdf=pdf / pdf.size,
        dense=True,
        diag_shift=self.diag_shift,
        diag_scale=self.diag_scale,
        holomorphic=self.holomorphic,
        chunk_size=chunk_size,
    )
    # distributed_eigh builds the dense QGT already row-sharded + padded
    if self.distributed_eigh:
        Sd = jax.block_until_ready(sharded_to_dense(self._S))
    else:
        Sd = jax.block_until_ready(_qgt_to_dense(self._S))
    self._step_times["qgt"] += time.perf_counter() - _ts

    # --- timed section: eigendecomposition + regularized linear solve ---
    _ts = time.perf_counter()
    (
        self._loss_stats,
        self._dw,
        self._rmd,
        self._snr,
        self._snr_F,
        self._ev,
        self._ev_reg,
    ) = jax.block_until_ready(
        _solve_dense(
            state.parameters,
            state.n_samples,
            E_loc,
            self._S,
            Sd,
            importance_weights.reshape(samples_q.shape[:-1]),
            self._loss_grad_factor,
            self.rcond,
            self.rcond_smooth,
            self.snr_atol,
            self.distributed_eigh,
        )
    )
    self._step_times["solve"] += time.perf_counter() - _ts

    self._info = make_monitor_dict(
        self._rmd, ess, self._snr, self._snr_F, self._ev, self._ev_reg
    )
    if stage == 0:
        # First stage of a (re)attempted step: start a fresh per-stage record.
        self._info_stages = {}
        self._last_qgt = self._S
    self._info_stages[stage] = self._info

    return self._dw


@jax.jit
def ess_from_weights(w):
    s1_sq = jnp.mean(w, axis=0) ** 2
    s2 = jnp.mean(w**2, axis=0)
    # Return normalized ESS in [0, 1]
    return (s1_sq / (s2 + jnp.finfo(w.dtype).eps)).squeeze()


@jax.jit
def ess_from_weights_var(w):
    # sum over the sample axis

    s1_sq = jnp.mean(w, axis=0) ** 2
    s2 = jnp.mean(w**2, axis=0)
    return ((s1_sq / (s2 - s1_sq + jnp.finfo(w.dtype).eps))).squeeze()