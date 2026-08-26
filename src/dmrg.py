"""Loading of exact MPS reference correlations and comparison metrics.

MPS data lives under
``<root>/<topology>_<shape>_precision<prec>/<t_a>ns/chi<D>/correlations_uppertriangular_20_seeds.npz``
with a single array ``corrs`` of shape ``(n_instances, n_pairs)`` holding <Z_i Z_j>
for i < j (row-major upper triangle), one row per instance/seed.

The instances and correlations folders use different topology names for some
geometries (e.g. instances ``3ddimer``/``biclique`` vs. correlations ``3d``/``rbm``);
:data:`CORR_TOPOLOGY` maps from the instances-folder name to the correlations one.
"""

import os
import glob
import numpy as np


# instances-folder topology name -> correlations-folder topology name
CORR_TOPOLOGY = {"2d": "2d", "3ddimer": "3d", "biclique": "rbm", "diamond": "diamond"}


def _corr_folder(root, topology, shape, t_a, precision):
    corr_topo = CORR_TOPOLOGY.get(topology, topology)
    return os.path.join(
        root, f"{corr_topo}_{tuple(shape)}_precision{precision}", f"{t_a}ns"
    )


def available_t_a(root, topology, shape, precision=256):
    """Sorted list of annealing times (ns) with MPS correlations for this instance."""
    corr_topo = CORR_TOPOLOGY.get(topology, topology)
    base = os.path.join(root, f"{corr_topo}_{tuple(shape)}_precision{precision}")
    return sorted(
        int(os.path.basename(d)[:-2]) for d in glob.glob(os.path.join(base, "*ns"))
    )


def load_mps_corrs(root, topology, shape, t_a, precision=256):
    """Return MPS correlations at the largest available bond dimension (chi).

    Returns an ``(n_instances, n_pairs)`` array.
    """
    folder = _corr_folder(root, topology, shape, t_a, precision)
    chi_dirs = glob.glob(os.path.join(folder, "chi*"))
    if not chi_dirs:
        raise FileNotFoundError(f"No MPS correlations under {folder}")
    best = chi_dirs[int(np.argmax([int(os.path.basename(d)[3:]) for d in chi_dirs]))]
    npz = os.path.join(best, "correlations_uppertriangular_20_seeds.npz")
    return np.load(npz)["corrs"]


def load_qpu_corrs(root, topology, shape, t_a, precision=256, processor=None):
    """Return D-Wave QPU correlations for this instance/t_a.

    QPU data mirrors the MPS layout but lives one level deeper, under a
    per-processor-generation folder and with no ``chi`` sub-directory (the QPU
    has no bond dimension)::

        <root>/<processor>/<topology>_<shape>_precision<prec>/<t_a>ns/correlations_uppertriangular_20_seeds.npz

    with a single ``(n_instances, n_pairs)`` array ``corrs`` (same convention as
    :func:`load_mps_corrs`). ``processor`` selects the D-Wave generation (e.g.
    ``"adv1"``/``"adv2"``); if None, the newest generation with data for this
    geometry is used (processor folders sorted descending, so ``adv2`` before
    ``adv1``).

    Returns an ``(n_instances, n_pairs)`` array.
    """
    corr_topo = CORR_TOPOLOGY.get(topology, topology)
    leaf = os.path.join(
        f"{corr_topo}_{tuple(shape)}_precision{precision}", f"{t_a}ns",
        "correlations_uppertriangular_20_seeds.npz",
    )
    if processor is not None:
        processors = [processor]
    else:
        processors = sorted(
            (os.path.basename(d) for d in glob.glob(os.path.join(root, "*"))
             if os.path.isdir(d)),
            reverse=True,  # prefer the newer generation (adv2 before adv1)
        )
    for proc in processors:
        npz = os.path.join(root, proc, leaf)
        if os.path.exists(npz):
            return np.load(npz)["corrs"]
    raise FileNotFoundError(
        f"No QPU correlations for {corr_topo} {tuple(shape)} at {t_a}ns under {root}"
        + (f"/{processor}" if processor else "")
    )


def correlation_error(X, Y):
    """Relative RMS error between correlation vectors, per instance (axis=-1).

    ``||X - Y||_2 / ||Y||_2`` over correlation pairs.
    """
    X = np.atleast_2d(X)
    Y = np.atleast_2d(Y)
    num = np.sqrt(np.mean((X - Y) ** 2, axis=-1))
    den = np.sqrt(np.mean(Y ** 2, axis=-1))
    return num / den
