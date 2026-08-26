"""Exact time-dependent TFIM quantum-anneal dynamics for small spin glasses."""

import os

import numpy as np
import netket as nk
import scipy.sparse.linalg as ssla
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Instance loading
# ---------------------------------------------------------------------------
def instance_dir_name(topology, shape, precision):
    """Folder name for an instance set, e.g. ``2d_(4, 4)_precision256``.

    The ``tuple`` repr matches the on-disk naming under ``data_dwave/instances``.
    """
    return f"{topology}_{tuple(shape)}_precision{precision}"


def load_instance(root, topology, shape, instance, precision=256):
    """Load one spin-glass instance from ``data_dwave/instances``.

    Reads ``<root>/<instance_dir_name>/seed{NN}.npz`` (arrays ``i``, ``j``,
    ``Jij``).

    Returns
    -------
    weights : (n_edges,) ndarray of coupling strengths J_ij.
    edges : list of (i, j) tuples (file order; the physics is order-independent).
    """
    path = os.path.join(
        root, instance_dir_name(topology, shape, precision), f"seed{instance:02d}.npz"
    )
    d = np.load(path)
    edges = list(zip(d["i"].tolist(), d["j"].tolist()))
    return np.asarray(d["Jij"], dtype=float), edges


def coupling_matrix(L, weights, edges):
    """Per-edge coupling matrix ``Jmat`` of shape ``(n_edges, L)``.

    Row ``n`` holds ``weights[n]`` in the two columns of edge ``n``. Used by the
    weighted Ising operator (:class:`src.operator.SpinGlassIsing`).
    """
    Jmat = np.zeros((len(edges), L))
    for n, (i, j) in enumerate(edges):
        Jmat[n, i] = weights[n]
        Jmat[n, j] = weights[n]
    return Jmat


# ---------------------------------------------------------------------------
# Annealing schedule
# ---------------------------------------------------------------------------
def load_schedule(path):
    """Load ``qa_schedule.csv`` (space-separated ``s  Gamma(s)  J(s)``, no header).

    Returns ``(s, gamma, J)`` arrays with ``s`` in [0, 1] and strengths in GHz.
    """
    data = np.loadtxt(path)
    s, gamma, J = data[:, 0], data[:, 1], data[:, 2]
    return s, gamma, J


def schedule_interpolators(path):
    """Return ``(s, f_gamma, f_J)`` where ``f_*`` interpolate over ``s in [0, 1]``."""
    s, gamma, J = load_schedule(path)
    f_gamma = interp1d(s, gamma, fill_value="extrapolate")
    f_J = interp1d(s, J, fill_value="extrapolate")
    return s, f_gamma, f_J


# ---------------------------------------------------------------------------
# Hamiltonian terms (netket IsingJax -> scipy sparse)
# ---------------------------------------------------------------------------
def build_terms(L, weights, edges):
    """Return ``(Hzz, Hx)`` scipy sparse matrices for an ``L``-site system.

    ``Hzz`` is the weighted spin-glass coupling term and ``Hx = -sum_i sigma^x_i``.
    """
    hi = nk.hilbert.Spin(0.5, N=L)

    # Weighted ZZ term: sum a single-edge IsingJax (J=1, no field) per bond,
    # scaled by the bond weight.
    Hzz = None
    for (i, j), w in zip(edges, weights):
        op = nk.operator.IsingJax(hi, graph=[[int(i), int(j)]], h=0.0, J=1.0)
        term = w * op.to_sparse()
        Hzz = term if Hzz is None else Hzz + term

    # Transverse field: h=1, J=0 gives exactly -sum_i sigma^x_i over all sites.
    # (Edges are irrelevant when J=0, but a non-empty graph is required.)
    Hx = nk.operator.IsingJax(hi, graph=[[0, 1]], h=1.0, J=0.0).to_sparse()

    return Hzz.tocsr(), Hx.tocsr()


def ground_state(H):
    """Lowest-eigenvalue eigenvector of a sparse Hermitian ``H`` (returns a vector)."""
    if H.shape[0] <= 2:
        w, v = np.linalg.eigh(H.toarray())
        return v[:, 0]
    _, v = ssla.eigsh(H.astype(complex), k=1, which="SA")
    return v[:, 0]


# ---------------------------------------------------------------------------
# Observables (read directly from the statevector)
# ---------------------------------------------------------------------------
def spin_configs(L):
    """(2**L, L) array of +/-1 spin configurations in netket basis order.

    The ordering matches ``build_terms``,
    so spin configurations can be read straight off this basis.
    """
    return np.asarray(nk.hilbert.Spin(0.5, N=L).all_states())


def zz_correlations(psi, configs):
    """Flat array of <sigma^z_i sigma^z_j> for all i < j (row-major upper triangle).

    ``psi`` is a statevector, ``configs`` the (2**L, L) +/-1 basis from
    :func:`spin_configs`.
    """
    psi = np.asarray(psi).reshape(-1)
    p = np.abs(psi) ** 2
    L = configs.shape[1]
    C = (configs * p[:, None]).T @ configs  # <Z_i Z_j>, shape (L, L)
    return np.real(C[np.triu_indices(L, k=1)])