#!/usr/bin/env python
"""Rewind t-VMC runs to one of their ``state_t*.nk`` checkpoints, selected by config.

Runs are named by the same YAML configs ``run_tvmc.py`` takes, and their directory is
derived with :func:`src.utils.get_save_path` (called with ``create=False`` here, so a
config that never ran is reported rather than given an empty folder).

``run_tvmc.py`` resumes from two files in that directory:

  * ``log.json``  -- the serialized ``nk.logging.RuntimeLog``; resume reads the LAST
    entry of ``t``, ``step`` and ``dt`` from it (see run_tvmc.py:241-250).
  * ``state.nk``  -- the variational state, loaded by ``CheckpointCallback.restore_state``.

So rewinding means: truncate every history in ``log.json`` to ``iters <= t_target`` (the
histories are indexed by ``t``, not by step count) and put the matching
``state_t{t_target:.3f}.nk`` in place of ``state.nk``. This does both, keeps backups, and
verifies the result deserializes.

Usage
-----
    # one config, several configs, or a whole experiment folder
    python revert_run.py configs/final_20/biclique_2x7x7_t_a20_rank4_instance0.yaml --list
    python revert_run.py configs/final_20/biclique_2x7x7_t_a20_rank4_instance0.yaml --to 0.35
    python revert_run.py configs/final_20 --to-latest --only-errored
    python revert_run.py configs/final_20/*.yaml --to 0.35 --apply --clean-dumps

    # escape hatch when you have a path but no config
    python revert_run.py --run-dir data/tvmc/.../e62a80f6 --to 0.35

Nothing is written without ``--apply``; the default is a dry run that prints the plan for
every selected run. ``--to-latest`` picks, per run, the newest checkpoint strictly before
that run's current last ``t`` -- the useful mode in bulk, since crashed jobs stop at
different times. ``--only-errored`` restricts to runs that have a ``tdvp_error_step*.json``.

Two guards are on by default, because both failure modes destroy good results silently:
runs marked ``done`` are skipped (they may have errored early and still finished --
``--include-done`` overrides), and a run whose ``log.json`` was written in the last
``--active-minutes`` (default 10) is treated as still running and skipped (``--force``
overrides). Rewinding under a live job loses the rewind, since it checkpoints on top.

``--clean-dumps`` moves the diagnostics of the discarded steps (``tdvp_error_step*.json``,
``tdvp_recovered_step*.json``, ``tdvp_samples_step*.npz``) into
``<RUN_DIR>/reverted_t{...}_{stamp}/`` so a later crash is not confused with the old one.
"""

import argparse
import glob
import json
import os
import re
import shutil
import sys
import time

import numpy as np
import yaml

from src.utils import get_save_path

STATE_PAT = re.compile(r"state_t(\d+\.\d+)\.nk$")
DUMP_PATS = ("tdvp_error_step*.json", "tdvp_recovered_step*.json", "tdvp_samples_step*.npz")
STEP_IN_NAME = re.compile(r"step(\d+)")


# --------------------------------------------------------------------- config -> run dir
def collect_config_paths(items):
    """Expand CLI arguments into a sorted list of YAML config paths.

    Accepts individual ``.yaml`` files and directories (all ``*.yaml`` inside, one level).
    """
    out = []
    for it in items:
        if os.path.isdir(it):
            found = sorted(glob.glob(os.path.join(it, "*.yaml")))
            if not found:
                print(f"[warn] no *.yaml in {it}")
            out.extend(found)
        elif os.path.isfile(it):
            out.append(it)
        else:
            print(f"[warn] no such config: {it}")
    return sorted(dict.fromkeys(out))


def resolve_runs(config_paths):
    """[(label, run_dir, exists)] for each config, via ``get_save_path(create=False)``."""
    runs = []
    for p in config_paths:
        with open(p) as f:
            config = yaml.safe_load(f)
        try:
            run_dir = get_save_path(config, create=False)
        except (KeyError, ValueError) as e:
            print(f"[warn] {p}: cannot derive save_path ({e})")
            continue
        runs.append((os.path.relpath(p), run_dir,
                     os.path.isfile(os.path.join(run_dir, "log.json"))))
    return runs


# --------------------------------------------------------------------------- log helpers
def find_checkpoints(run_dir):
    """{t: path} for every ``state_t*.nk`` in the run dir, sorted by t."""
    out = {}
    for p in glob.glob(os.path.join(run_dir, "state_t*.nk")):
        m = STATE_PAT.search(os.path.basename(p))
        if m:
            out[float(m.group(1))] = p
    return dict(sorted(out.items()))


def history_iters(entry):
    """The ``iters`` list of a serialized History, or None if this key has none."""
    if isinstance(entry, dict) and isinstance(entry.get("iters"), list):
        return entry["iters"]
    return None


def truncate(node, n_orig, keep):
    """Slice every list of length ``n_orig`` under ``node`` down to the ``keep`` indices.

    A serialized History is ``{"iters": [...], "value": ...}``, but ``value`` is not always
    a flat list: MCStats histories nest one list per field (``Mean``, ``Variance``,
    ``R_hat``, ...), complex values are split into ``{"real": [...], "imag": [...]}``, and
    vector-valued keys (``pt_accept_per_beta``, ``0/ev``) hold a list per step. Matching on
    length covers all of those without hardcoding shapes.
    """
    if isinstance(node, dict):
        return {k: truncate(v, n_orig, keep) for k, v in node.items()}
    if isinstance(node, list):
        return [node[i] for i in keep] if len(node) == n_orig else node
    return node


def log_tail(log):
    """(n_points, last t, last step, last dt, done) of a loaded log.json."""
    def last(key, cast=float):
        e = log.get(key)
        if history_iters(e) is None:
            return None
        try:
            return cast(np.ravel(np.asarray(e.get("value"), float))[-1])
        except (TypeError, ValueError):
            return None
    return (len(history_iters(log.get("t")) or []), last("t"), last("step", int),
            last("dt"), "done" in log)


# --------------------------------------------------------------------------- one run
def revert_one(label, run_dir, args):
    """Plan (and with --apply, perform) the rewind of a single run. Returns a status str."""
    log_path = os.path.join(run_dir, "log.json")
    with open(log_path) as f:
        log = json.load(f)
    ckpts = find_checkpoints(run_dir)
    n, t_last, step_last, dt_last, done = log_tail(log)
    errored = sorted(glob.glob(os.path.join(run_dir, "tdvp_error_step*.json")))
    age_min = (time.time() - os.path.getmtime(log_path)) / 60.0

    print(f"\n=== {label}")
    print(f"    {run_dir}")
    print(f"    log: {n} points, last t = {t_last}, step = {step_last}, dt = {dt_last}, "
          f"done = {done}, error dumps = {len(errored)}, written {age_min:.1f} min ago")
    print(f"    checkpoints: " + (", ".join(f"{t:.3f}" for t in ckpts) or "none"))

    if args.list:
        return "listed"
    # A live job holds its own state in memory and checkpoints on top of whatever we
    # write, so rewinding underneath it silently loses the rewind (or worse, mixes them).
    if age_min < args.active_minutes and not args.force:
        print(f"    skipped: log.json written {age_min:.1f} min ago, job is probably still "
              f"running (--force, or --active-minutes, to override)")
        return "skipped"
    # `--only-errored` matches runs that errored at ANY point, including ones that
    # recovered and finished; rewinding those would throw away good results.
    if done and not args.include_done:
        print("    skipped: run is marked done (--include-done to rewind it anyway)")
        return "skipped"
    if args.only_errored and not errored:
        print("    skipped: no tdvp_error_step*.json (--only-errored)")
        return "skipped"

    # ---- choose the target t
    if args.to_latest:
        earlier = [t for t in ckpts if t_last is None or t < t_last - 1e-12]
        if not earlier:
            print(f"    skipped: no checkpoint before t = {t_last}")
            return "skipped"
        t_target = earlier[-1]
    else:
        match = [t for t in ckpts if np.isclose(t, args.to, atol=1e-9)]
        if not match and not args.keep_state:
            print(f"    skipped: no state_t{args.to:.3f}.nk")
            return "skipped"
        t_target = match[0] if match else args.to

    # ---- where to cut
    iters = history_iters(log.get("t"))
    if iters is None:
        print("    skipped: log.json has no 't' history")
        return "skipped"
    it = np.ravel(np.asarray(iters, float))
    keep = [i for i, v in enumerate(it) if v <= t_target + 1e-12]
    if not keep:
        print(f"    skipped: nothing at t <= {t_target} (log starts at {it.min()})")
        return "skipped"
    n_orig, last_kept_t = it.size, float(it[keep[-1]])

    last_kept_step = None
    step_hist = log.get("step")
    if history_iters(step_hist) is not None:
        sv = np.ravel(np.asarray(step_hist["value"], float))
        if sv.size == n_orig:
            last_kept_step = int(sv[keep[-1]])

    print(f"    plan: keep {len(keep)}/{n_orig} points (drop {n_orig - len(keep)}); "
          f"cut at t <= {t_target:.6f}")
    print(f"          last kept t = {last_kept_t:.6f} "
          f"(gap to checkpoint {t_target - last_kept_t:+.2e}), step = {last_kept_step}")
    if not args.keep_state:
        print(f"          state.nk <- state_t{t_target:.3f}.nk")
    if done:
        print("          drop the 'done' flag so the run is resumable")

    dumps = []
    if args.clean_dumps and last_kept_step is not None:
        for pat in DUMP_PATS:
            for p in glob.glob(os.path.join(run_dir, pat)):
                m = STEP_IN_NAME.search(os.path.basename(p))
                if m and int(m.group(1)) > last_kept_step:
                    dumps.append(p)
        print(f"          quarantine {len(dumps)} diagnostic file(s) from steps "
              f"> {last_kept_step}")

    if not args.apply:
        return "planned"

    stamp = time.strftime("%Y%m%d-%H%M%S")
    # 1. log.json
    shutil.copy2(log_path, f"{log_path}.bak-{stamp}")
    new = {}
    for key, entry in log.items():
        if key == "done":
            continue
        e_it = history_iters(entry)
        if e_it is None:
            new[key] = entry                     # not a History: pass through untouched
        elif len(e_it) != n_orig:
            # logged on a different cadence (the stage keys miss the pre-step point):
            # cut it on its own iters rather than the 't' index
            own = np.ravel(np.asarray(e_it, float))
            new[key] = truncate(entry, own.size,
                                [i for i, v in enumerate(own) if v <= t_target + 1e-12])
        else:
            new[key] = truncate(entry, n_orig, keep)
    with open(log_path, "w") as f:
        json.dump(new, f)
    print(f"    wrote log.json (backup log.json.bak-{stamp})")

    # 2. state.nk
    if not args.keep_state:
        state = os.path.join(run_dir, "state.nk")
        if os.path.isfile(state):
            shutil.copy2(state, f"{state}.bak-{stamp}")
        shutil.copy2(ckpts[t_target], state)
        print(f"    wrote state.nk <- state_t{t_target:.3f}.nk "
              f"(backup state.nk.bak-{stamp})")

    # 3. quarantine stale diagnostics
    if dumps:
        dest = os.path.join(run_dir, f"reverted_t{t_target:.3f}_{stamp}")
        os.makedirs(dest, exist_ok=True)
        for p in dumps:
            shutil.move(p, os.path.join(dest, os.path.basename(p)))
        print(f"    moved {len(dumps)} diagnostic file(s) -> {os.path.basename(dest)}")

    # 4. verify through the same code path run_tvmc.py uses
    try:
        import netket as nk
        lg = nk.logging.RuntimeLog.deserialize(os.path.join(run_dir, "log"))
        t0 = float(np.ravel(lg["t"].to_dict()["value"])[-1])
        step0 = int(np.ravel(lg["step"].to_dict()["value"])[-1])
        dt0 = float(np.ravel(lg["dt"].to_dict()["value"])[-1])
        try:
            fin = bool(lg["done"])
        except KeyError:
            fin = False
        print(f"    verified: resume at t = {t0:.6f}, step = {step0}, dt = {dt0:.6g}, "
              f"done = {fin}")
        if not np.isclose(t0, last_kept_t):
            print(f"    WARNING: expected last t = {last_kept_t:.6f}")
    except Exception as e:
        print(f"    WARNING: could not verify with netket ({e!r}); log.json was still "
              f"rewritten and backed up")
    return "reverted"


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="*",
                    help="YAML config(s) as passed to run_tvmc.py, and/or directories of them")
    ap.add_argument("--run-dir", action="append", default=[],
                    help="operate on this run dir directly (repeatable); skips config lookup")
    ap.add_argument("--to", type=float, help="target t; needs a state_t{t:.3f}.nk")
    ap.add_argument("--to-latest", action="store_true",
                    help="per run, the newest checkpoint strictly before its current last t")
    ap.add_argument("--list", action="store_true", help="list checkpoints and exit")
    ap.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    ap.add_argument("--only-errored", action="store_true",
                    help="only touch runs that have a tdvp_error_step*.json")
    ap.add_argument("--include-done", action="store_true",
                    help="also rewind runs marked done (skipped by default)")
    ap.add_argument("--active-minutes", type=float, default=10.0,
                    help="treat a run whose log.json is newer than this as live and skip it "
                         "(default 10)")
    ap.add_argument("--force", action="store_true",
                    help="rewind even a run that looks live")
    ap.add_argument("--clean-dumps", action="store_true",
                    help="move diagnostics of discarded steps into reverted_t{...}/")
    ap.add_argument("--keep-state", action="store_true",
                    help="only truncate log.json, leave state.nk alone")
    args = ap.parse_args()

    if not args.configs and not args.run_dir:
        ap.error("give at least one config (or --run-dir)")
    if not args.list and args.to is None and not args.to_latest:
        ap.error("need --to T, --to-latest, or --list")
    if args.to is not None and args.to_latest:
        ap.error("--to and --to-latest are mutually exclusive")

    runs = resolve_runs(collect_config_paths(args.configs))
    runs += [(f"(--run-dir) {os.path.basename(d.rstrip('/'))}", d.rstrip("/"),
              os.path.isfile(os.path.join(d.rstrip("/"), "log.json")))
             for d in args.run_dir]
    if not runs:
        sys.exit("nothing to do")

    missing = [(lab, d) for lab, d, ok in runs if not ok]
    live = [(lab, d) for lab, d, ok in runs if ok]
    print(f"{len(runs)} config(s)/run(s): {len(live)} with a log.json, {len(missing)} without")
    for lab, d in missing:
        print(f"  [no log.json] {lab}\n                {d}")

    status = {}
    for lab, d in live:
        try:
            status[lab] = revert_one(lab, d, args)
        except Exception as e:                       # one bad run must not abort the batch
            print(f"    ERROR: {e!r}")
            status[lab] = "error"

    counts = {}
    for v in status.values():
        counts[v] = counts.get(v, 0) + 1
    print("\nsummary: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if not args.apply and not args.list:
        print("dry run -- nothing written. re-run with --apply")
    elif args.apply:
        print("Note: wandb resumes by config hash, so replayed steps append to the "
              "existing wandb run rather than overwrite it.")


if __name__ == "__main__":
    main()
