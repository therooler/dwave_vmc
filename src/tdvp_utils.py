import jax
import jax.numpy as jnp
import netket.jax as nkjax
import jax.scipy as jsp
from functools import partial
from netket.utils.types import Array
from netket.operator import AbstractOperator

@jax.jit
def make_monitor_dict(epsilon_squared, ess, snr, snr_F, ev, ev_reg):
    """
    Build a small diagnostics dict for logging / monitoring.

    Args
    ----
    epsilon_squared    : scalar         # TDVP residual r^2(t)
    ess                : scalar         # effective sample size from blurred weights
    snr                : (n_params,)    # per-eigendirection force SNR (eigenbasis of S)
    snr_F              : (n_params,)    # SNR of the force estimator (parameter basis)
    ev                 : (n_params,)    # Eigenvalues of QGT
    ev_reg             : (n_params,)    # regularized (effective) eigenvalues

    Returns
    -------
    metrics : dict of scalars (JAX arrays)
    """

    # Clean SNRs: replace inf/NaN with 0 for summary stats
    def _clean(x):
        x = jnp.where(jnp.isfinite(x), x, 0.0)
        return x

    snr_clean = _clean(snr)
    snrF_clean = _clean(snr_F)
    ev_clean = _clean(ev)
    ev_reg_clean = _clean(ev_reg)

    # Eigenbasis SNR summaries
    snr_min = jnp.min(snr_clean)
    snr_med = jnp.median(snr_clean)
    snr_sorted = jnp.sort(snr_clean)
    idx_10p = jnp.maximum(0, (snr_sorted.shape[0] * 10) // 100)
    snr_10p = snr_sorted[idx_10p]

    # Parameter-basis (force) SNR summaries
    snrF_min = jnp.min(snrF_clean)
    snrF_med = jnp.median(snrF_clean)

    metrics = {
        "epsilon_squared": epsilon_squared,
        "ess_blurred": ess,
        "snr_min": snr_min,
        "snr_10p": snr_10p,
        "snr_med": snr_med,
        "snrF_min": snrF_min,
        "snrF_med": snrF_med,
        "snr": snr,
        "snr_F": snr_F,
        "ev": ev_clean,
        "ev_reg": ev_reg_clean,
    }
    return metrics


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
    # jax.debug.print("w {} s1_sq {} s2 {}",w, s1_sq, s2 )
    return ((s1_sq / (s2 - s1_sq + jnp.finfo(w.dtype).eps))).squeeze()


def _logpsi_eloc_single(x_p, params, apply_fn, op):
    """For a single configuration ``x_p``: return ``(E_loc, logpsi_stay, logpsi_all)``"""
    x_p = x_p.reshape(-1)
    x_p_conn, mels = op.get_conn_padded(x_p)
    logpsi_stay = apply_fn({"params": params}, x_p)
    logpsi_all = apply_fn({"params": params}, x_p_conn)
    E_loc = jnp.sum(
        mels * jnp.exp(logpsi_all - jnp.expand_dims(logpsi_stay, -1)), axis=-1
    )
    return jnp.atleast_1d(E_loc), logpsi_stay, logpsi_all


@partial(jax.jit, static_argnames=("apply_fn", "chunk_size"))
def reweight_on_configs(x, params, apply_fn, op, chunk_size):
    """Local energy and ``log psi`` at ``params`` for the *fixed* configurations ``x``."""
    x_shape = x.shape
    x = x.reshape(-1, x_shape[-1])

    def _f(_x):
        E_loc, logpsi_stay, _ = _logpsi_eloc_single(_x, params, apply_fn, op)
        return E_loc, logpsi_stay

    vf = jax.vmap(_f, in_axes=0)
    if chunk_size is None:
        return vf(x)
    return nkjax.apply_chunked(
        vf, in_axes=0, chunk_size=chunk_size, axis_0_is_sharded=False
    )(x)


@partial(
    jax.jit, static_argnames=("apply_fn", "chunk_size", "diagonal_mels")
)
def blurred_sample(
    x: Array,
    key,
    params,
    q: float,
    apply_fn,
    op: AbstractOperator,
    chunk_size,
    diagonal_mels: bool = True,
):
    """One-step "blurred" proposal with importance weights.

    For each input configuration ``x[i]``, this kernel constructs a simple mixture proposal:

    - with probability ``q`` it keeps the configuration unchanged;
    - with probability ``1-q`` it proposes a *single* random connected configuration sampled
      uniformly from ``op.get_conn_padded(x[i])``.

    The returned scalar weight ``w_blurred`` corrects expectations from this mixture proposal to
    the target density :math:`p(\sigma) \propto |\psi(\sigma)|^2` (computed from
    ``apply_fn({'params': params}, ·).real``).

    Parameters
    ----------
    x:
        Array of shape ``(batch, n_dof)`` (or generally ``(batch, ...)``) containing the input
        configurations.
    key:
        JAX PRNGKey.
    params:
        Parameters passed to ``apply_fn``.
    q:
        Mixture parameter in ``[0, 1]`` controlling the probability of *staying* at the current
        configuration.
    apply_fn:
        Callable such that ``apply_fn({'params': params}, x)`` returns ``log(psi(x))`` (possibly
        complex). Only the real part is used to form :math:`|\psi|^2`.
    op:
        Operator providing ``get_conn_padded`` returning connected configurations and matrix
        elements.
    chunk_size:
        If not ``None``, evaluates the per-sample function with ``nkjax.apply_chunked``.

    Returns
    -------
    x_p:
        Array with the same shape as ``x`` containing the proposed (or unchanged) configurations.
    w_blurred:
        Array of shape ``(batch,)`` with importance weights
        :math:`w = p_{\mathrm{target}}(x_p) / p_{\mathrm{mix}}(x_p)`, where
        :math:`p_{\mathrm{target}}(\sigma) \propto |\psi(\sigma)|^2` and
        :math:`p_{\mathrm{mix}}(\sigma) = q\,p_{\mathrm{target}}(\sigma) + (1-q)\,\frac{1}{n}\sum_j p_{\mathrm{target}}(\sigma_j)`.
    E_loc:
        Local energy estimate for each proposed configuration ``x_p[i]``.
    logpsi_stay:
        ``log psi(x_p)`` at ``params`` for each proposed configuration (shape ``(batch,)``). This
        is the reference amplitude needed to reweight these cached configurations to new parameters
        (see :func:`reweight_on_configs`).
    """
    x_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    batch_size = x.shape[0]
    # rng for u1, u2 per configuration
    c = jax.random.uniform(key, shape=(batch_size, 2))

    def get_blurred_sample_and_Eloc(_in):
        _x, rng = _in
        u1, u2 = rng
        _x_shape = _x.shape
        _x = _x.reshape(-1)
        # Connected elements of Hamiltonian
        x_conn, _ = op.get_conn_padded(_x)
        # NOTE: get_conn_padded(_x) can contain diagonal elements, which correspond to "stay" configuration
        # For Ising, the first element will be diagonal, we therefore only have nconn-1 off-diagonal elements
        n_conn = x_conn.shape[-2] - 1
        idx = jnp.floor(u2 * n_conn).astype(jnp.int32)
        # Only choose from off-diagonal elements
        proposed = x_conn[idx + 1]
        # choose a whether to flip or stay
        x_p = jnp.where(u1 > q, _x, proposed)  # equivalent to u1 < 1-q
        # log |psi| (stay + neighbors) and the local energy on the proposed config
        E_loc, logpsi_stay, logpsi_all = _logpsi_eloc_single(x_p, params, apply_fn, op)
        # Off-diagonal connection count (padding is constant across configs).
        n_conn = logpsi_all.shape[-1] - 1
        # target density ∝ |psi|^2
        logp_stay = 2.0 * logpsi_stay.real
        logp_all = 2.0 * logpsi_all.real  # (n,)
        # stable mixture weight: (1-q)*p(stay) + (q/n)*sum_j p(all_flipped_j)
        log_term_main = jnp.log1p(-q) + logp_stay
        log_term_flips = (
            jnp.log(q) - jnp.log(n_conn) + jsp.special.logsumexp(logp_all[1:])
        )
        log_w_blurred = jsp.special.logsumexp(
            jnp.stack([log_term_main, log_term_flips])
        )
        w_blurred = jnp.exp(logp_stay - log_w_blurred)  # scalar
        return x_p.reshape(_x_shape), w_blurred, E_loc, logpsi_stay

    vmapped_get_blurred_sample_and_weight = jax.vmap(
        get_blurred_sample_and_Eloc, in_axes=0
    )
    if chunk_size is None:
        out = vmapped_get_blurred_sample_and_weight((x, c))
    else:
        out = nkjax.apply_chunked(
            vmapped_get_blurred_sample_and_weight,
            in_axes=0,
            chunk_size=chunk_size,
            axis_0_is_sharded=False,
        )((x, c))
    return out[0].reshape(x_shape), out[1], out[2], out[3]

