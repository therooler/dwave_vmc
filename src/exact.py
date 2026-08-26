"""Exact time-dependent TFIM quantum-anneal dynamics for small spin glasses.

Everything needed to evolve a 2D spin-glass instance through the D-Wave
annealing schedule and read off its 2-local ZZ correlations:

  * instance loading           -> load_instance / instance_dir_name
  * schedule loading           -> load_schedule / schedule_interpolators
  * Hamiltonian terms          -> build_terms (netket IsingJax -> sparse) / ground_state
  * observables                -> spin_configs / zz_correlations
  * exact evolution            -> run_quench (QuTiP)

The transverse-field Ising Hamiltonian for the anneal is

    H(s) = J(s) * Hzz  +  Gamma(s) * Hx ,
    Hzz = sum_<ij> w_ij sigma^z_i sigma^z_j  (weighted spin glass),
    Hx  = - sum_i sigma^x_i                  (transverse field).

With anneal parameter s in [0, 1] traversed in physical time t_a (ns), and after
the change of integration variable t = t_a * s, the Schrodinger equation that is
integrated reads

    i d|psi>/ds = pi * t_a * [ J(s) Hzz + Gamma(s) Hx ] |psi> ,   s : 0 -> 1 ,

starting from the ground state of H(s=0) (the transverse-field paramagnet). The
factor pi converts the GHz schedule to the angular-frequency / Pauli convention
used to generate the DMRG reference data.
"""

import os

import numpy as np
import netket as nk
import qutip
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

    The ordering matches ``build_terms`` (and the QuTiP states evolved from it),
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


# ---------------------------------------------------------------------------
# Exact evolution
# ---------------------------------------------------------------------------
def run_quench(topology, shape, instance, t_a, schedule_path, instance_root,
               precision=256, n_out=200, return_states=False):
    """Run the exact anneal for one instance and return final-time ZZ correlations.

    Parameters
    ----------
    topology, shape : instance identifiers (e.g. ``"2d"``, ``[4, 4]``); select the
        folder under ``instance_root``.
    instance : integer instance seed.
    t_a : annealing time in ns.
    schedule_path : path to ``qa_schedule.csv``.
    instance_root : root of the instances dataset (``data_dwave/instances``).
    precision : coupling precision (256 or 1), must match the reference dataset.
    n_out : number of output time points along s in [0, 1].
    return_states : if True also return ``(s_out, states)``.

    Returns
    -------
    corrs : flat array of <Z_i Z_j>, i < j, at s = 1.
    """
    weights, edges = load_instance(instance_root, topology, shape, instance, precision)
    L = max(max(i, j) for i, j in edges) + 1
    Hzz, Hx = build_terms(L, weights, edges)
    _, f_gamma, f_J = schedule_interpolators(schedule_path)

    # Initial state: ground state of H(s=0).
    H0 = float(f_J(0.0)) * Hzz + float(f_gamma(0.0)) * Hx
    psi0 = qutip.Qobj(ground_state(H0).reshape(-1, 1))

    prefactor = np.pi * t_a
    H = [
        [qutip.Qobj(Hzz), lambda s, **kw: prefactor * float(f_J(s))],
        [qutip.Qobj(Hx), lambda s, **kw: prefactor * float(f_gamma(s))],
    ]

    s_out = np.linspace(0.0, 1.0, n_out)
    result = qutip.sesolve(H, psi0, s_out, e_ops=[])

    configs = spin_configs(L)
    corrs = zz_correlations(result.states[-1].full(), configs)

    if return_states:
        return corrs, s_out, result.states
    return corrs
