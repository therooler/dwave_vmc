"""Generate the t-VMC YAML configs for the paper's methodological experiments.

Each `###` header in ``paper_experiments.md`` becomes its own subfolder under
``configs_final/`` (one experiment per folder), and every config in it is the shared
paper default (``PAPER_BASE``) with a single axis varied. ``disbatch.sh`` can then run
each subfolder like any other experiment. Usage::

    python make_paper_configs.py            # write configs_final/<experiment>/*.yaml
    python make_paper_configs.py --clean    # wipe configs_final/ first

Mirrors ``make_configs.py`` conventions (flat YAML, ``_fmt`` filenames, config-hash
resumability) so ``run_tvmc.py`` / ``verify_tvmc.py`` consume the output unchanged.

The ``mixed_precision`` experiment emits a ``compute_dtype`` key, which
``run_tvmc.build_state`` forwards to ``CorrelatorJastrow``.
"""

import argparse
import ast
import glob
import os
import shutil

import yaml

from src.utils import n_sites_from_instance

# Shared default setup for every paper experiment (see paper_experiments.md "Default settings"):
# CorrelatorJastrow [2,4], 2^16 samples, 512 chains with parallel tempering, rcond_smooth=1e-12,
# single instance (instance 0) at t_a=7ns. Remaining keys match make_configs.BASE.
PAPER_BASE = dict(
    precision=256,
    schedule="data_dwave/qa_schedule.csv",
    model="jastrow",
    orders=[2, 4],
    rank=1,
    seed=100,
    instance=0,
    t_a=7,
    n_samples=2**17,
    n_chains=2048,
    sampler="pt",
    pt_n_replicas=16,
    n_vmc_steps=1000,
    vmc_lr=1e-3,
    diag_shift=1e-5,
    q=0.3,
    rcond=1e-14,
    rcond_smooth=1e-12,
    snr_atol=1,
    integrator="rk45",
    dt_max=1.0e-2,
    dt_min=1.0e-4,
    cache_within_step=True,
    distributed_eigh=False,
    n_save_times=21,
    every_n_steps=25,
    wandb=True,
    data_dir="./data",
    instance_dir="./data_dwave/instances",
    corr_dir="./data_dwave/correlations/mps",
)

# Largest on-disk shape per topology (data_dwave/instances/), used by the rank experiment.
LARGEST = {
    "biclique": [2, 9, 9],
    "diamond": [4, 4, 8],
    "3ddimer": [3, 3, 3],
    "2d": [8, 8],
}
# Minimal ranks required to get <5%
MIN_RANK = {
    7: {
        ("diamond", (3, 3, 8)): 1,
        ("diamond", (4, 4, 8)): 1,
        ("2d", (4, 4)): 1,
        ("2d", (5, 5)): 1,
        ("2d", (6, 6)): 1,
        ("2d", (7, 7)): 1,
        ("2d", (8, 8)): 2,
        ("biclique", (2, 5, 5)): 1,
        ("biclique", (2, 6, 6)): 2,
        ("biclique", (2, 7, 7)): 3,
        ("biclique", (2, 8, 8)): 2,
        ("biclique", (2, 9, 9)): 3,
        ("3ddimer", (3, 2, 2)): 1,
        ("3ddimer", (3, 2, 3)): 2,
        ("3ddimer", (3, 3, 3)): 1,
    },
    20: {
        ("diamond", (3, 3, 8)): 1,
        ("diamond", (4, 4, 8)): 2,
        ("2d", (4, 4)): 1,
        ("2d", (5, 5)): 2,
        ("2d", (6, 6)): 3,
        ("2d", (7, 7)): 3,
        ("2d", (8, 8)): 3,
        ("biclique", (2, 5, 5)): 1,
        ("biclique", (2, 6, 6)): 2,
        ("biclique", (2, 7, 7)): 3,
        ("biclique", (2, 8, 8)): 2,
        ("biclique", (2, 9, 9)): 4,
        ("3ddimer", (3, 2, 2)): 2,
        ("3ddimer", (3, 2, 3)): 3,
        ("3ddimer", (3, 3, 3)): 3,
    },
}

# Disorder-seed instances for the final production runs.
N_INSTANCES = 5

# The large-N multi-GPU case and is generated separately (distributed_biclique).
FINAL_EXCLUDE = {("biclique", (2, 18, 18))}


def available_shapes(instance_dir, topology, precision=256):
    """All on-disk shapes for a topology, parsed from instance folder names.

    Instance folders are named ``{topology}_{tuple(shape)}_precision{precision}``
    (e.g. ``3ddimer_(3, 2, 2)_precision256``). Returns a list of shape lists,
    sorted small-to-large.
    """
    prefix, suffix = f"{topology}_", f"_precision{precision}"
    shapes = []
    for d in glob.glob(os.path.join(instance_dir, f"{topology}_*{suffix}")):
        name = os.path.basename(d)
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        try:
            shapes.append(list(ast.literal_eval(name[len(prefix) : -len(suffix)])))
        except (ValueError, SyntaxError):
            continue
    return sorted(shapes, key=lambda s: (len(s), s))


def _fmt(v):
    """Filename-safe rendering of a value (no spaces/brackets -> safe as a CLI arg)."""
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, (list, tuple)):
        return "-".join(_fmt(x) for x in v)
    if isinstance(v, float):
        return f"{v:g}".replace(".", "p").replace("-", "m")
    return str(v)


def _cfg(topology, shape, **overrides):
    """Build one config dict off PAPER_BASE with topology/shape and swept overrides."""
    return dict(topology=topology, shape=list(shape), **overrides)


def build_experiments():
    """Return a list of experiment specs: dict(folder, tag_keys, configs)."""
    experiments = []

    # 1. Local sampling versus parallel tempering: diamond [4,4,8], R=3.
    lvp_cfgs = [_cfg("biclique", [2, 6, 6], rank=3, sampler=s) for s in ("local", "pt")]
    experiments.append(
        dict(folder="local_vs_pt", tag_keys=["sampler"], configs=lvp_cfgs)
    )

    # 2. Cached vs uncached RK45: 3ddimer [3,2,2], toggle cache_within_step.
    rk45_cfgs = [
        _cfg("biclique", [2, 6, 6], rank=3, cache_within_step=c) for c in (True, False)
    ]
    experiments.append(
        dict(folder="cached_rk45", tag_keys=["cache_within_step"], configs=rk45_cfgs)
    )

    # 3. Blurred sampling: biclique [2,9,9], R=3, sweep the blur strength q = 0.0 vs 0.3.
    blur_cfgs = [_cfg("biclique", [2, 6, 6], rank=3, q=q) for q in (0.0, 0.3)]
    experiments.append(
        dict(folder="blurred_sampling", tag_keys=["q"], configs=blur_cfgs)
    )

    # 4. Final production runs: every available shape of every topology, swept over
    # rank R=1..4 at a single instance (the rank-vs-error scan). For each (topology,
    # size) the minimal rank that clears 5% at this t_a (MIN_RANK[t_a]) is additionally
    # run at N_INSTANCES seeds for production statistics; a size absent from MIN_RANK
    # (no reference data) stays at 1 instance for every rank.
    for t_a in [7, 20]:
        final_cfgs = []
        for topo in LARGEST.keys():
            for shape in available_shapes(
                PAPER_BASE["instance_dir"], topo, PAPER_BASE["precision"]
            ):
                if (topo, tuple(shape)) in FINAL_EXCLUDE:
                    continue
                min_rank = MIN_RANK[t_a].get((topo, tuple(shape)))
                for rank in [1, 2, 3, 4]:
                    n_inst = N_INSTANCES if rank == min_rank else 1
                    for i in range(n_inst):
                        final_cfgs.append(
                            _cfg(
                                topo,
                                shape,
                                rank=rank,
                                orders=[2, 4],
                                instance=i,
                                t_a=t_a,
                            )
                        )
        experiments.append(
            dict(
                folder=f"final_{t_a}",
                tag_keys=["t_a", "rank", "instance"],
                configs=final_cfgs,
            )
        )

    experiments.append(
        dict(
            folder="biclique_ns",
            tag_keys=["n_samples"],
            configs=[
                _cfg(
                    "biclique",
                    [2, 18, 18],
                    rank=2,
                    n_samples=n_samples,
                    legacy_rank_k2=False,  # remove redundant parameters.
                )
                for n_samples in [2**17, 2**18,]
            ],
        )
    )

    experiments.append(
        dict(
            folder="large_diamond",
            tag_keys=["rank"],
            configs=[
                _cfg(
                    "diamond",
                    [8, 8, 8],
                    rank=1,
                    legacy_rank_k2=False,  # remove redundant parameters.
                )
            ],
        )
    )
    return experiments


def main():
    parser = argparse.ArgumentParser(
        description="generate paper t-VMC experiment configs"
    )
    parser.add_argument("--configs_root", default="configs")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="remove the configs_final root before generating",
    )
    args = parser.parse_args()

    if args.clean and os.path.isdir(args.configs_root):
        shutil.rmtree(args.configs_root)

    n = 0
    for exp in build_experiments():
        out_dir = os.path.join(args.configs_root, exp["folder"])
        os.makedirs(out_dir, exist_ok=True)
        for overrides in exp["configs"]:
            topology = overrides["topology"]
            shape = overrides["shape"]
            config = {
                **PAPER_BASE,
                **overrides,
                "experiment_name": exp["folder"],
            }
            # sweep_size = true spin count N, read from the instance edge list (shape
            # is only a folder id and does not equal N -- e.g. 3ddimer has N = 2*prod(shape)).
            n_sites = n_sites_from_instance(
                config["instance_dir"],
                topology,
                shape,
                config["instance"],
                config["precision"],
            )
            config.setdefault("sweep_size", 4 * n_sites)

            size_tag = f"{topology}_" + "x".join(map(str, shape))
            tag = size_tag + "".join(f"_{k}{_fmt(config[k])}" for k in exp["tag_keys"])
            path = os.path.join(out_dir, f"{tag}.yaml")
            with open(path, "w") as f:
                f.write(
                    "# auto-generated by make_paper_configs.py "
                    f"(experiment: {exp['folder']})\n"
                )
                yaml.safe_dump(config, f, sort_keys=False)
            n += 1
            print(f"[INFO] wrote {path}")

    print(f"[INFO] generated {n} config(s) under {args.configs_root}")


if __name__ == "__main__":
    main()
