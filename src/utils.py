"""Save-path construction for t-VMC runs."""

import os
import json
import hashlib

import numpy as np


def get_save_path(config, create=True):
    """Return a unique save path for a t-VMC run's data and checkpoints.

    The path is hierarchical and human-readable, ending in an 8-hex digest of the
    full config so that any change to the configuration maps to a distinct folder
    (mirroring the wandb run-id trick). ``run_tvmc.py`` and ``verify_tvmc.py`` call
    this with the same config so they agree on the location.

    Parameters
    ----------
    config : dict of run parameters.
    create : if True, create the directory.
    """
    experiment_name = config.get("experiment_name", None)
    if experiment_name is None:
        raise ValueError("`experiment_name` cannot be empty")
    data_dir = config.get("data_dir", "./data")
    topology, shape = config["topology"], config["shape"]
    instance = config["instance"]
    t_a = config["t_a"]
    rank = config.get("rank", 1)
    model_name = config.get("model", "jastrow")
    if model_name == "rbm":
        model_str = f"rbm_a{config.get('alpha', 1)}"
    else:
        orders = config.get("orders", [1, 2, 4])
        model_str = "jastrow_o" + "".join(map(str, orders))
        if model_name == "jastrow_plaquette":
            cycle_size = config.get("cycle_size", 4)
            # "_plaq" keeps the default 4-cycle tag; other sizes get "_plaq6" etc.
            model_str += "_plaq" + ("" if cycle_size == 4 else str(cycle_size))
        if rank > 1:
            model_str += f"_r{rank}"
    # Schmitt force-SNR filter tag (only when enabled, so runs without it keep their
    # existing paths). The full-config hash below also distinguishes the value.
    snr_atol = config.get("snr_atol")
    if snr_atol is not None:
        model_str += f"_snr{snr_atol}"
    q = config.get("q", 0.0)
    n_samples = config["n_samples"]
    n_chains = config["n_chains"]
    sweep_size = config["sweep_size"]
    integrator = config.get("integrator", "rk45")
    dt_min = config.get("dt_min", 1e-4)
    dt_max = config.get("dt_max", 1e-2)
    sampler = config.get("sampler", "local")
    if sampler == "exact":
        sampler_str = "exact"
    elif sampler == "pt":
        sampler_str = f"pt_r{config.get('pt_n_replicas', 16)}"
    else:
        sampler_str = "local"
    cfg_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:8]

    path = os.path.join(
        data_dir,
        "tvmc",
        experiment_name,
        f"{topology}_" + "x".join(map(str, shape)),
        f"instance_{instance}",
        f"t_a_{t_a}",
        model_str,
        f"{sampler_str}_ns_{n_samples}_nc_{n_chains}_sw_{sweep_size}",
        f"q{q}",
        f"{integrator}_dt_min{dt_min:.2e}_dt_max{dt_max:.2e}",
        cfg_hash,
    )
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def n_sites_from_instance(instance_dir, topology, shape, instance=0, precision=256):
    """True spin count ``N`` for an instance, read from its edge list on disk.

    ``shape`` names the on-disk instance folder but is NOT the number of spins:
    the 3ddimer dimer geometry has two sites per unit cell (so ``N = 2*prod(shape)``),
    while biclique goes the other way. ``N`` is the number of distinct node indices
    in the edge list, i.e. ``max(i, j) + 1`` -- matching how ``run_tvmc.build_generator``
    derives ``L``. Used by the config generators to set ``sweep_size`` to the real ``N``.

    The folder-name format mirrors ``src.exact.instance_dir_name`` (kept inline here so
    this stays a numpy-only helper and the config generators avoid the netket/qutip import).
    """
    fname = f"{topology}_{tuple(shape)}_precision{precision}"
    path = os.path.join(instance_dir, fname, f"seed{instance:02d}.npz")
    d = np.load(path)
    return int(max(int(d["i"].max()), int(d["j"].max())) + 1)


def enumerate_cycles(edges, cycle_size):
    """All simple cycles of length ``cycle_size`` in the graph, as sorted node tuples.

    ``edges`` is a list of ``(i, j)`` node-index pairs (as returned by
    ``src.exact.load_instance``). Returns a tuple of sorted ``cycle_size``-tuples, one
    per distinct cycle, deterministically ordered.

    For the ansatz each cycle contributes a monomial ``prod_{i in cycle} sigma_i`` that
    depends only on the *set* of its nodes, so cycles are deduplicated by the
    ``frozenset`` of their nodes (two cycles on the same node set give the same monomial
    and must not be double-counted). Cycles are enumerated by DFS of simple paths rooted
    at their minimum node (every subsequent node strictly greater than the root), which
    reaches each cycle once per traversal direction; the frozenset dedup collapses the two
    directions. Suitable for the small, sparse instance graphs here (N <= 64, degree <= 5);
    cost grows like ``N * deg**(cycle_size - 1)``.

    Examples (``cycle_size=4``): 3ddimer 3x3x3 -> 81, 2d 8x8 -> 56, diamond 4x4x8 -> 0
    (diamond is girth-6, so its shortest cycles are hexagons, ``cycle_size=6``).
    """
    if cycle_size < 3:
        raise ValueError(f"cycle_size must be >= 3, got {cycle_size}")
    adj = {}
    for i, j in edges:
        adj.setdefault(i, set()).add(j)
        adj.setdefault(j, set()).add(i)
    found = set()

    def dfs(root, current, path, visited):
        if len(path) == cycle_size:
            if root in adj[current]:  # closes back to the root -> a cycle
                found.add(frozenset(path))
            return
        for nb in adj[current]:
            if nb > root and nb not in visited:
                visited.add(nb)
                path.append(nb)
                dfs(root, nb, path, visited)
                path.pop()
                visited.remove(nb)

    for root in sorted(adj):
        dfs(root, root, [root], {root})
    return tuple(sorted(tuple(sorted(s)) for s in found))


def enumerate_4cycles(edges):
    """All 4-cycles (plaquettes); thin wrapper over :func:`enumerate_cycles`."""
    return enumerate_cycles(edges, 4)
