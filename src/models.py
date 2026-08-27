"""Variational ansatze for the t-VMC quench."""

from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import flax.linen as nn

from typing import Any

from jax.nn.initializers import normal

from netket.utils.types import NNInitFunc
from netket import nn as nknn

default_kernel_init = normal(stddev=0.01)


class CorrelatorJastrow(nn.Module):
    r"""Jastrow-Feenberg correlator ansatz.

    For spins :math:`\sigma_i = \pm 1` the log-amplitude is a sum of k-body
    correlators

    .. math::
        \log\psi(\sigma) = \sum_{k\in\text{orders}} \mathcal J^{(k)}(\sigma),
        \quad
        \mathcal J^{(k)}(\sigma) = \sum_{i_1<\dots<i_k}
            W^{(k)}_{i_1\dots i_k}\,\sigma_{i_1}\cdots\sigma_{i_k},

    with the rank-2 factorization of each tensor (Eq. 8), one strictly-upper-
    triangular complex matrix :math:`U^{(k)}` per order (nonzero only for
    :math:`i<j`),

    .. math::
        W^{(k)}_{i_1\dots i_k}
            = U^{(k)}_{i_1 i_2} U^{(k)}_{i_2 i_3}\cdots U^{(k)}_{i_{k-1} i_k}.

    The ordered-tuple sum collapses to a transfer-matrix recursion of ``k-1``
    matrix products, giving an :math:`O(N^2)` evaluation per order: with ``g``
    initialized to ``sigma`` (shape ``(B, N)``), repeating ``g = sigma * (g @ U)``
    exactly ``k-1`` times and summing over the last axis yields
    :math:`\mathcal J^{(k)}`. Each matmul appends one strictly-larger index, so
    after ``k-1`` steps ``g_j`` sums all chains ``i_1<\dots<i_{k-1}<j``.

    Only the :math:`N(N-1)/2` strict-upper-triangle couplings are free parameters.
    ``U`` is built from a flat vector by the *fill-triangular* reshape trick:
    ``triu(reshape(concat([v, v[::-1], zeros(N)]), (N, N)), 1)`` lands each entry of
    ``v`` in exactly one strict-upper position (the duplicated copies fall in the
    masked lower triangle/diagonal).

    Parameters are complex (holomorphic) and initialized to small complex-normal
    noise scaled by ``1/N`` so ``log psi ~ 0`` (the uniform H(s=0) paramagnet).

    Setting ``rank > 1`` sums ``rank`` independent chain-products per order,
    ``\mathcal J^{(k)} = \sum_{r=1}^{R}\sum_{i_1<\dots<i_k}
    U^{(k,r)}_{i_1 i_2}\cdots U^{(k,r)}_{i_{k-1}i_k}`` -- i.e. a rank-2R
    factorization of ``W^{(k)}`` instead of rank-2. For k=2 the
    increased rank is linear, hence R=1 always there.
    """

    orders: Sequence[int] = (2, 4)
    rank: int = 1
    legacy_rank_k2: bool = False
    """Keep the redundant ``rank`` axis on the ``k=2`` correlator (legacy).
    Only set to true for reproducing the original experiments!!!
    """
    param_dtype: type = complex
    param_initializer: NNInitFunc = default_kernel_init
    compute_dtype: Any = None
    """Precision of the forward pass: jnp.float64, jnp.float32, jnp.float16, jnp.float8
    Splits Im-> Re-Im for compute_dtype<jnp.float32.
    """

    # native JAX complex dtype for each real float width that has one
    _native_complex = {"float32": jnp.complex64, "float64": jnp.complex128}

    def _stacked(self, k):
        """Whether ``params[k]`` carries a leading ``rank`` axis.

        ``k=1`` never does (its param is the ``(N,)`` vector ``W1``); ``k=2`` only
        does under ``legacy_rank_k2``, the rank axis being redundant there.
        """
        if k == 1 or self.rank == 1:
            return False
        return k > 2 or self.legacy_rank_k2

    @nn.compact
    def __call__(self, x):
        batch_shape = x.shape[:-1]
        N = x.shape[-1]
        n_upper = N * (N - 1) // 2

        # Declare all params in param_dtype (complex) up front so the parameter pytree does
        # not depend on ``compute_dtype`1
        params = {}
        for k in sorted(self.orders):
            if k < 1:
                raise ValueError(f"order must be >= 1, got {k}")
            if k == 1:
                params[k] = self.param(
                    "W1", self.param_initializer, (N,), self.param_dtype
                )
            elif not self._stacked(k):
                params[k] = self.param(
                    f"V{k}", self.param_initializer, (n_upper,), self.param_dtype
                )
            else:
                params[k] = self.param(
                    f"V{k}",
                    self.param_initializer,
                    (self.rank, n_upper),
                    self.param_dtype,
                )

        if self.compute_dtype is None:
            out = self._forward_complex(x, N, params, jnp.dtype(self.param_dtype))
        else:
            cdt = jnp.dtype(self.compute_dtype)
            native = self._native_complex.get(cdt.name)
            if native is not None:
                out = self._forward_complex(x, N, params, jnp.dtype(native))
            else:
                out = self._forward_split(x, N, params, cdt)

        return out.reshape(batch_shape)

    def _forward_complex(self, x, N, params, cdt):
        """Original chain math evaluated in the native complex dtype ``cdt``.

        ``cdt == complex128`` (``compute_dtype`` None/float64) reproduces the production
        path exactly; ``complex64`` (float32) is genuine float32 arithmetic. Casting the
        (complex128) params/inputs down to ``cdt`` is what reduces the precision.
        """
        sigma = x.reshape((-1, N)).astype(cdt)  # (B, N)

        def chain(v, k):
            v = v.astype(cdt)
            W = jnp.concatenate([v, v[::-1], jnp.zeros(N, cdt)])
            U = jnp.triu(W.reshape(N, N), 1)
            g = sigma
            for _ in range(k - 1):
                g = sigma * (g @ U)
            return jnp.sum(g, axis=-1)

        out = jnp.zeros((sigma.shape[0],), dtype=cdt)
        for k in sorted(self.orders):
            p = params[k].astype(cdt)
            if k == 1:
                out = out + sigma @ p
            elif not self._stacked(k):
                out = out + chain(p, k)
            else:
                out = out + jnp.sum(jax.vmap(lambda vr: chain(vr, k))(p), axis=0)
        return out

    def _forward_split(self, x, N, params, fdt):
        """Complex chain carried as two real arrays (real/imag) in low dtype ``fdt``.

        For ``float16``/``bfloat16``/``float8`` JAX has no native complex type, so every
        product is done with real ``fdt`` matmuls: ``(ar+i ai)(br+i bi)`` becomes
        ``(ar@br - ai@bi) + i(ar@bi + ai@br)``.
        """
        sr = x.reshape((-1, N)).astype(fdt)  # sigma real part (+/-1, exact in fdt)
        si = jnp.zeros_like(sr)  # sigma imag part = 0

        def cmatmul(ar, ai, br, bi):
            return ar @ br - ai @ bi, ar @ bi + ai @ br

        zeros_N = jnp.zeros(N, fdt)

        def triu_fill(vv):
            W = jnp.concatenate([vv, vv[::-1], zeros_N])
            return jnp.triu(W.reshape(N, N), 1)

        def chain(v, k):
            Ur = triu_fill(v.real.astype(fdt))
            Ui = triu_fill(v.imag.astype(fdt))
            gr, gi = sr, si
            for _ in range(k - 1):
                tr, ti = cmatmul(gr, gi, Ur, Ui)  # g @ U
                gr, gi = sr * tr, sr * ti  # sigma * (g@U); sigma imag is 0
            return jnp.sum(gr, axis=-1), jnp.sum(gi, axis=-1)

        out_r = jnp.zeros((sr.shape[0],), fdt)
        out_i = jnp.zeros((sr.shape[0],), fdt)
        for k in sorted(self.orders):
            p = params[k]
            if k == 1:
                out_r = out_r + sr @ p.real.astype(fdt)  # si=0 -> imag is only sr@bi
                out_i = out_i + sr @ p.imag.astype(fdt)
            elif not self._stacked(k):
                cr, ci = chain(p, k)
                out_r, out_i = out_r + cr, out_i + ci
            else:
                crs, cis = jax.vmap(lambda vr: chain(vr, k))(p)
                out_r = out_r + jnp.sum(crs, axis=0)
                out_i = out_i + jnp.sum(cis, axis=0)
        # lax.complex requires >= float32 components; the heavy matmuls already ran in fdt.
        return jax.lax.complex(out_r.astype(jnp.float32), out_i.astype(jnp.float32))


class RBM(nn.Module):
    r"""A restricted boltzman Machine, equivalent to a 2-layer FFNN with a
    nonlinear activation function in between.
    """

    param_dtype: Any = complex
    """The dtype of the weights."""
    activation: Any = nknn.activation.log_cosh
    """The nonlinear activation function."""
    alpha: float | int = 1
    """feature density. Number of features equal to alpha * input.shape[-1]"""
    use_hidden_bias: bool = True
    """if True uses a bias in the dense layer (hidden layer bias)."""
    use_visible_bias: bool = True
    """if True adds a bias to the input not passed through the nonlinear layer."""
    precision: Any = None
    """numerical precision of the computation see :class:`jax.lax.Precision` for details."""

    kernel_init: NNInitFunc = default_kernel_init
    """Initializer for the Dense layer matrix."""
    hidden_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the hidden bias."""
    visible_bias_init: NNInitFunc = default_kernel_init
    """Initializer for the visible bias."""

    compute_dtype: Any = None
    """Precision of the forward pass: jnp.float64, jnp.float32, jnp.float16, jnp.float8.
    Params stay in ``param_dtype`` (complex); the matmuls are cast down to
    ``compute_dtype``. For widths with no native complex type (float16/bfloat16/float8)
    the linear layer is carried as split real/imag matmuls, mirroring CorrelatorJastrow.
    """

    # native JAX complex dtype for each real float width that has one
    _native_complex = {"float32": jnp.complex64, "float64": jnp.complex128}

    @nn.compact
    def __call__(self, input):
        batch_shape = input.shape[:-1]
        N = input.shape[-1]
        inp = input.reshape((-1, N))
        features = int(self.alpha * N)

        # Declare params in param_dtype (complex) up front so the parameter pytree does
        # not depend on ``compute_dtype`` (same convention as CorrelatorJastrow).
        kernel = self.param("kernel", self.kernel_init, (N, features), self.param_dtype)
        hidden_bias = (
            self.param("hidden_bias", self.hidden_bias_init, (features,), self.param_dtype)
            if self.use_hidden_bias
            else None
        )
        visible_bias = (
            self.param("visible_bias", self.visible_bias_init, (N,), self.param_dtype)
            if self.use_visible_bias
            else None
        )
        params = (kernel, hidden_bias, visible_bias)

        if self.compute_dtype is None:
            out = self._forward_complex(inp, params, jnp.dtype(self.param_dtype))
        else:
            cdt = jnp.dtype(self.compute_dtype)
            native = self._native_complex.get(cdt.name)
            if native is not None:
                out = self._forward_complex(inp, params, jnp.dtype(native))
            else:
                out = self._forward_split(inp, params, cdt)

        return out.reshape(batch_shape)

    def _forward_complex(self, inp, params, cdt):
        """RBM forward evaluated in the native complex dtype ``cdt``.

        ``cdt == complex128`` (``compute_dtype`` None/float64) reproduces the production
        path exactly; ``complex64`` (float32) is genuine float32 arithmetic. Casting the
        (complex128) params/inputs down to ``cdt`` is what reduces the precision.
        """
        kernel, hidden_bias, visible_bias = params
        sigma = inp.astype(cdt)  # real +/-1 promoted to complex (imag 0)
        theta = jnp.matmul(sigma, kernel.astype(cdt), precision=self.precision)
        if hidden_bias is not None:
            theta = theta + hidden_bias.astype(cdt)
        out = jnp.sum(self.activation(theta), axis=-1)
        if visible_bias is not None:
            out = out + jnp.matmul(
                sigma, visible_bias.astype(cdt), precision=self.precision
            )
        return out

    def _forward_split(self, inp, params, fdt):
        """Linear layer carried as split real/imag matmuls in low dtype ``fdt``.

        For ``float16``/``bfloat16``/``float8`` JAX has no native complex type, so the
        (expensive) matmuls run as real ``fdt`` products; ``sigma`` is real (imag 0), so
        ``sigma @ (kr + i ki) = sigma@kr + i sigma@ki``. The complex is reassembled at
        float32 (``lax.complex`` requires >= float32 components) before the transcendental
        ``log_cosh`` and the reduction.
        """
        kernel, hidden_bias, visible_bias = params
        sr = inp.astype(fdt)  # real +/-1, exact in fdt

        theta_r = jnp.matmul(sr, kernel.real.astype(fdt), precision=self.precision)
        theta_i = jnp.matmul(sr, kernel.imag.astype(fdt), precision=self.precision)
        if hidden_bias is not None:
            theta_r = theta_r + hidden_bias.real.astype(fdt)
            theta_i = theta_i + hidden_bias.imag.astype(fdt)

        theta = jax.lax.complex(theta_r.astype(jnp.float32), theta_i.astype(jnp.float32))
        out = jnp.sum(self.activation(theta), axis=-1)

        if visible_bias is not None:
            br = jnp.matmul(sr, visible_bias.real.astype(fdt), precision=self.precision)
            bi = jnp.matmul(sr, visible_bias.imag.astype(fdt), precision=self.precision)
            out = out + jax.lax.complex(br.astype(jnp.float32), bi.astype(jnp.float32))
        return out


class JastrowRBM(nn.Module):
    r"""Product ansatz :math:`\psi = \psi_\text{Jastrow}\,\psi_\text{RBM}`, i.e. the
    log-amplitudes add:

    .. math::
        \log\psi(\sigma) = \mathcal J^{[\text{orders}]}(\sigma) + \text{RBM}(\sigma).

    The Jastrow factor supplies the exact, cheap low-order (2-/4-body) ZZ structure; the
    RBM factor's ``log cosh`` hidden units add a *compact, all-order* correction (implicitly
    every body-order, packed into :math:`O(N\cdot\alpha N)` learned weights) for the local
    high-order and sign/phase structure a truncated Jastrow cannot reach -- e.g. the dense
    3D loop web of the 3ddimer topology.

    Both factors are the existing :class:`CorrelatorJastrow` / :class:`RBM` modules used
    verbatim as submodules (no re-implemented chain math or log-cosh), so mixed-precision
    (``compute_dtype``) and the complex holomorphic parameterization are inherited unchanged.
    Parameters live under the ``jastrow`` and ``rbm`` submodule scopes.
    """

    orders: Sequence[int] = (2, 4)
    rank: int = 1
    legacy_rank_k2: bool = True
    alpha: float | int = 1
    param_dtype: type = complex
    param_initializer: NNInitFunc = default_kernel_init
    compute_dtype: Any = None

    @nn.compact
    def __call__(self, x):
        jastrow = CorrelatorJastrow(
            orders=self.orders,
            rank=self.rank,
            legacy_rank_k2=self.legacy_rank_k2,
            param_dtype=self.param_dtype,
            param_initializer=self.param_initializer,
            compute_dtype=self.compute_dtype,
            name="jastrow",
        )
        rbm = RBM(
            alpha=self.alpha,
            param_dtype=self.param_dtype,
            compute_dtype=self.compute_dtype,
            name="rbm",
        )
        return jastrow(x) + rbm(x)


class PlaquetteCorrelator(nn.Module):
    r"""Graph-native cycle-body correlator: one free complex weight per cycle.

    .. math::
        \log\psi_\text{plaq}(\sigma) = \sum_{p\in\text{plaquettes}}
            c_p\,\prod_{i\in p}\sigma_i,

    where ``plaquettes`` lists the length-``k`` cycles of the instance graph (see
    :func:`src.utils.enumerate_cycles`); every entry has the same length ``k`` (4 for
    plaquettes, 6 for hexagons, ...). Unlike :class:`CorrelatorJastrow`'s *democratic*
    order-``k`` term over all index tuples, this places genuine ``k``-body monomials
    exactly on the graph's frustrated loops -- ``P`` complex params (``P = 81`` for 3ddimer
    3x3x3 at ``k=4``). Each monomial is a product of ``k`` :math:`\pm 1` spins, hence
    exactly :math:`\pm 1`, so it needs no fan-in normalization; the tangent norm per param
    is O(1).
    """

    plaquettes: tuple = ()
    """Tuple of equal-length node-index tuples, one per cycle (static, hashable field)."""
    param_dtype: type = complex
    param_initializer: NNInitFunc = default_kernel_init
    compute_dtype: Any = None

    # native JAX complex dtype for each real float width that has one
    _native_complex = {"float32": jnp.complex64, "float64": jnp.complex128}

    @nn.compact
    def __call__(self, x):
        batch_shape = x.shape[:-1]
        N = x.shape[-1]
        P = len(self.plaquettes)
        if P == 0:
            return jnp.zeros(batch_shape, dtype=jnp.dtype(self.param_dtype))

        c = self.param("c", self.param_initializer, (P,), self.param_dtype)
        idx = jnp.asarray(self.plaquettes, dtype=jnp.int32)  # (P, 4)
        sigma = x.reshape((-1, N))  # (B, N), +/-1

        if self.compute_dtype is None:
            out = self._forward_complex(sigma, idx, c, jnp.dtype(self.param_dtype))
        else:
            cdt = jnp.dtype(self.compute_dtype)
            native = self._native_complex.get(cdt.name)
            if native is not None:
                out = self._forward_complex(sigma, idx, c, jnp.dtype(native))
            else:
                out = self._forward_split(sigma, idx, c, cdt)
        return out.reshape(batch_shape)

    def _forward_complex(self, sigma, idx, c, cdt):
        # (B, P, 4) -> product over the four corners -> (B, P), exact +/-1
        mono = jnp.prod(sigma[:, idx], axis=-1).astype(cdt)
        return mono @ c.astype(cdt)

    def _forward_split(self, sigma, idx, c, fdt):
        """Real +/-1 monomials against split real/imag weights for widths with no
        native complex type (float16/bfloat16/float8). ``mono`` is real so
        ``mono @ (cr + i ci) = mono@cr + i mono@ci``; reassembled at float32."""
        mono = jnp.prod(sigma[:, idx], axis=-1).astype(fdt)
        out_r = mono @ c.real.astype(fdt)
        out_i = mono @ c.imag.astype(fdt)
        return jax.lax.complex(out_r.astype(jnp.float32), out_i.astype(jnp.float32))


class JastrowPlaquette(nn.Module):
    r"""Sum ansatz :math:`\log\psi = \mathcal J^{[\text{orders}]}(\sigma)
    + \log\psi_\text{plaq}(\sigma)`.

    The Jastrow factor supplies the cheap *democratic* low-order (2-/4-body) ZZ structure;
    the plaquette factor adds graph-native cycle-body terms on the instance's frustrated
    cycles (4-cycles for 3ddimer's dense 3D loop web; 6-cycles/hexagons for a girth-6 graph
    like diamond) -- the local high-order content a truncated chain-Jastrow cannot
    efficiently reach. Both are the existing
    :class:`CorrelatorJastrow` / :class:`PlaquetteCorrelator` modules used verbatim as
    submodules, so mixed precision (``compute_dtype``) and the complex holomorphic
    parameterization are inherited. Parameters live under the ``jastrow`` and ``plaquette``
    submodule scopes.
    """

    orders: Sequence[int] = (2, 4)
    rank: int = 1
    legacy_rank_k2: bool = True
    plaquettes: tuple = ()
    param_dtype: type = complex
    param_initializer: NNInitFunc = default_kernel_init
    compute_dtype: Any = None

    @nn.compact
    def __call__(self, x):
        jastrow = CorrelatorJastrow(
            orders=self.orders,
            rank=self.rank,
            legacy_rank_k2=self.legacy_rank_k2,
            param_dtype=self.param_dtype,
            param_initializer=self.param_initializer,
            compute_dtype=self.compute_dtype,
            name="jastrow",
        )
        plaq = PlaquetteCorrelator(
            plaquettes=self.plaquettes,
            param_dtype=self.param_dtype,
            param_initializer=self.param_initializer,
            compute_dtype=self.compute_dtype,
            name="plaquette",
        )
        return jastrow(x) + plaq(x)


def migrate_legacy_rank_k2(variables):
    """Fold a legacy ``(rank, n_upper)`` ``V2`` block down to ``(n_upper,)``.

    Maps parameters written by a ``legacy_rank_k2=True`` model onto the
    ``legacy_rank_k2=False`` parameterization of the *same* wavefunction. Exact,
    not approximate: the ``k=2`` correlator is linear in its parameters, so
    ``sum_r V2[r]`` reproduces ``log psi`` identically (see
    :class:`CorrelatorJastrow`).

    Accepts a bare params dict (``{"V2": ...}``), a variables dict
    (``{"params": {...}}``), or either with a ``"jastrow"`` sub-scope
    (:class:`JastrowRBM` / :class:`JastrowPlaquette`), and leaves every other leaf
    -- including ``V4`` and higher, whose rank axis is *not* redundant -- alone.
    Already-migrated (1-D ``V2``) trees pass through unchanged, so it is
    idempotent.

    Note that nothing validates parameter shapes at load time, so the result must
    be used with a model built with ``legacy_rank_k2=False``; pairing it with a
    legacy model fails later, inside the forward pass.
    """

    def fold(tree):
        # Mapping, not dict: flax's FrozenDict is not a dict subclass.
        if not isinstance(tree, Mapping):
            return tree
        out = {}
        for key, value in tree.items():
            if key == "V2" and jnp.ndim(value) == 2:
                out[key] = jnp.sum(value, axis=0)
            elif key in ("params", "jastrow"):
                out[key] = fold(value)
            else:
                out[key] = value
        return type(tree)(out) if not isinstance(tree, dict) else out

    return fold(variables)
