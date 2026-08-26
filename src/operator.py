"""A lean weighted transverse-field Ising operator"""

from functools import partial

import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

from netket.operator import DiscreteJaxOperator
from netket.operator._ising.base import IsingBase


@partial(jax.vmap, in_axes=(0, None, None))
def _ising_mels(x, h, J):
    """Matrix elements for one config: [diagonal ZZ, then -h for each spin flip]."""
    mask = J != 0
    Ezz = jnp.sum(J, axis=1) * (jnp.abs(x @ mask.T) - 1) / 2
    mels = jnp.empty((x.size + 1,), dtype=J.dtype)
    mels = mels.at[0].set(Ezz.sum())
    mels = mels.at[1:].set(-h)
    return mels


def _flip_if(cond, x, local_states):
    was_state_0 = x == local_states[0]
    s0 = jnp.asarray(local_states[0], dtype=x.dtype)
    s1 = jnp.asarray(local_states[1], dtype=x.dtype)
    return jnp.where(cond ^ was_state_0, s0, s1)


@partial(jax.vmap, in_axes=(0, None, None))
def _ising_conn_states(x, flip, local_states):
    return _flip_if(flip, x, local_states)


@register_pytree_node_class
class SpinGlassIsing(IsingBase, DiscreteJaxOperator):
    r"""Weighted TFIM ``sum_<ij> (J Jmat_ij) Z_i Z_j - h sum_i X_i`` (jax, diagonal mels)."""

    def __init__(self, hilbert, graph, h, J, Jmat, dtype=None):
        # IsingBase calls ``h.astype`` / ``J.astype`` so they must be arrays.
        super().__init__(hilbert, graph=graph, h=jnp.asarray(h), J=jnp.asarray(J),
                         dtype=dtype)
        self._h_jax = jnp.asarray(h, dtype=self.dtype)
        self._J_jax = jnp.asarray(J, dtype=self.dtype)
        self._Jmat = jnp.asarray(Jmat, dtype=self.dtype)
        self._flip = jnp.eye(self.max_conn_size, hilbert.size, k=-1, dtype=bool)
        self._local_states = tuple(hilbert.local_states)

    @property
    def max_conn_size(self) -> int:
        return self.hilbert.size + 1

    @jax.jit
    def get_conn_padded(self, x):
        batch_shape = x.shape[:-1]
        xr = x.reshape((-1, x.shape[-1]))
        mels = _ising_mels(xr, self._h_jax, self._J_jax * self._Jmat)
        mels = mels.reshape(batch_shape + mels.shape[1:])
        xp = _ising_conn_states(xr, self._flip, self._local_states)
        xp = xp.reshape(batch_shape + xp.shape[1:])
        return xp, mels

    def n_conn(self, x):
        return jnp.full(x.shape[:-1], self.max_conn_size, dtype=jnp.int32)

    def tree_flatten(self):
        data = (self._h_jax, self._J_jax, self._Jmat, self.edges)
        metadata = {"hilbert": self.hilbert, "dtype": self.dtype}
        return data, metadata

    @classmethod
    def tree_unflatten(cls, metadata, data):
        h, J, Jmat, edges = data
        return cls(metadata["hilbert"], edges, h=h, J=J, Jmat=Jmat, dtype=metadata["dtype"])
