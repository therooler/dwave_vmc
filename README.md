# Spin-glass TFIM quench: t-VMC

Code for *Numerical simulation of D-Wave's quantum advantage experiment with time-dependent variational Monte Carlo* [arxiv:2609.01719](https://arxiv.org/abs/2609.01719).

The data containing the checkpoints and final simulation data can be found at 10.5281/zenodo.22131893 (LINK WILL BE ADDED)

Simulate the time-dependent transverse-field Ising (TFIM) **quantum-anneal quench** of D-Wave
spin-glass instances following the schedule in [`data_dwave/qa_schedule.csv`](data_dwave/qa_schedule.csv), and
compare the resulting 2-local `⟨σᶻᵢσᶻⱼ⟩` correlations against MPS (DMRG) and QPU reference data.

With anneal parameter `s ∈ [0,1]` traversed in physical time `t_a` (ns), the state evolves under

$$
i d|ψ⟩/ds = π·t_a·[ J(s)·Hzz + Γ(s)·Hx ] |ψ⟩ , \qquad  H_{zz} = Σ_{<ij>} w_ij σᶻᵢσᶻⱼ ,   H_x = −Σᵢ σˣᵢ
$$

starting from the ground state of `H(0)`. The state is a Jastrow correlator
ansatz evolved with the blurred TDVP driver.

All reusable logic lives in [`src/`](src/); the root scripts/notebooks only orchestrate.

Please find the data and checkpoints of the paper here: **Zenodo LINK**
---

## Layout

```
data_dwave/
  qa_schedule.csv        D-Wave annealing schedule (s, Γ(s), J(s))
  instances/             spin-glass instances, per <topology>_<shape>_precision<prec>/
  correlations/mps/      MPS (DMRG) reference correlations, per instance / t_a / chi
  correlations/qpu/      D-Wave QPU reference correlations, per processor generation
configs/<experiment>/    generated run configs
data/tvmc/<experiment>/  run outputs: checkpoints + logs + corrs (see get_save_path)

src/
  exact.py        instance generation, schedule, Ising terms, exact QuTiP quench, ZZ correlations
  operator.py     SpinGlassIsing — weighted Ising jax operator 
  models.py       CorrelatorJastrow 
  tdvp_blurred.py TDVPBlurred driver 
  tdvp_utils.py   blurred-sampling / diagnostics helpers used by the driver
  dmrg.py         load MPS/QPU references + correlation_error metric
  utils.py        get_save_path — unique, hierarchical run path from a config
  logger.py       CheckpointCallback — resumable state+log checkpoints (+ optional wandb)
  callbacks.py    extra monitoring callbacks

make_configs.py        generate the experiments as YAML configs into configs/<experiment>/
run_tvmc.py            run one t-VMC quench from a YAML config 
revert_run.py          revert run to previous checkpoint
verify_tvmc.py         load a finished run, compute ZZ corrs, compare to MPS + QPU
figures.ipynb          paper figures, built from the verified runs on disk
benchmark_eigh         benchmark JAXMG solves.
```

## Setup

Use `uv`

```bash
python -m venv .venv && source .venv/bin/activate
uv sync
```

Alternatively, use

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The environment uses `netket`, `jax[cuda12]`, `qutip`, `nqxpack` (checkpoints) and `pyyaml`. Optional packages are `wandb` and `jaxmg`. 

---

## Workflow

### 1. Generate the experiments

[`make_configs.py`](make_configs.py) writes every run of every experiment as a flat YAML config.
Each config is the shared paper default (`PAPER_BASE`, at the top of the file) with a few keys
overridden, so an experiment is just "one axis varied off the default".

```bash
python make_configs.py                      # write configs/<experiment>/*.yaml
python make_configs.py --clean              # wipe configs/ first
```

The experiments themselves are defined in `build_experiments()`, one entry per folder:

| folder | what it varies |
|---|---|
| `local_vs_pt` | `sampler`: `local` vs `pt` (biclique [2,6,6], R=3) |
| `cached_rk45` | `cache_within_step`: `True` vs `False` (biclique [2,6,6], R=3) |
| `blurred_sampling` | blur strength `q`: `0.0` vs `0.3` (biclique [2,6,6], R=3) |
| `final_7`, `final_20` | production runs at `t_a` = 7 / 20 ns: every on-disk shape of every topology × rank `R=1..4` |
| `biclique_ns` | large-N biclique [2,18,18], `R=2` |
| `large_diamond` | large-N diamond [8x8x8], `R=1` |


Each config's filename tags the swept keys (`tag_keys`), e.g.
`configs/final_7/2d_8x8_t_a7_rank2_instance0.yaml`. Two details of the `final_*` sweep are worth
knowing: shapes come from whatever is on disk in `data_dwave/instances/` (`available_shapes`), and
for each (topology, size) the minimal rank that clears the 5% correlation-error target (`MIN_RANK`)
is run at `N_INSTANCES` disorder seeds instead of one, for production statistics. `sweep_size` is
set from the true spin count `N` read off the instance edge list — the shape is only a folder id
and does not equal `N` (a `3ddimer` [3,3,3] has `N = 2·27`).

To change what gets generated, edit `PAPER_BASE` (defaults), `MIN_RANK` / `N_INSTANCES` /
`FINAL_EXCLUDE` (production sweep), or add an entry to `build_experiments()`.

### 2. Run

```bash
python run_tvmc.py --config configs/final_7/2d_8x8_t_a7_rank2_instance0.yaml
```

`run_tvmc.py` prepares the `s=0` ground state via a short VMC, then evolves to `s=1` with
`TDVPBlurred`, checkpointing state + log to the run's `get_save_path` directory. It does **not**
compute correlations (that's verification, off the saved state).

Runs are **resumable**: re-running the same config restores the latest checkpoint and continues,
and exits immediately if that config already reached `s=1`. 

### 3. Verify

```bash
python verify_tvmc.py --config configs/final_7/2d_8x8_t_a7_rank2_instance0.yaml
python verify_tvmc.py --config … --t 1.0 --n-samples 1048576 --chunk-size 4096
```

Loads the final checkpointed state (or the `--t` snapshot, e.g. `--t 0.6`), sets `n_samples`,
computes all `N(N−1)/2` ZZ correlations once with their MC errorbars, and prints the relative
correlation error `eps_c` against the MPS reference and against the QPU reference (each skipped
with a message if no reference file exists for that instance/`t_a`).

Results are written next to the checkpoints as `corrs_n{n_samples}_t{s:.3f}.npz`.

### 4. Plot

All can be reproduced with `figures.ipynb`

> Note: for biclique 20ns instances, rare crashes can occur. Use `revert_run.py` to revert the run to a previous checkpoint to try again.
---

## Config reference

Key fields (see `make_configs.py` `PAPER_BASE` for all defaults):

| field | meaning |
|---|---|
| `legacy_rank_k2`| In a previous version of the code, there were redundant parameters for the 2-body correlators. **Set to False for new experiments that do not involve reproducing the original results!**|
| `topology`, `shape`, `instance`, `precision` | graph family, shape box, instance seed, coupling precision (256/1) |
| `t_a` | annealing time (ns); must match the reference file for comparison |
| `schedule` | path to `data_dwave/qa_schedule.csv` |
| `model` | ansatz: `jastrow` (correlator), `rbm`, `jastrow_rbm` |
| `orders`, `rank` | jastrow correlator orders and channels per order (rank-2R factorization) |
| `compute_dtype` | forward-pass precision (`float64`/`float32`/`float16`/…; default full precision) |
| `n_samples`, `n_chains`, `sweep_size` | MC sampling (sweep_size set from the true site count `N`) |
| `sampler` | `pt` (parallel tempering, default), `local`, or `exact` |
| `pt_n_replicas`, `pt_betas` | PT replicas per chain (even; β=1 physical) and β-ladder (`linear`/`log`/explicit) |
| `n_vmc_steps`, `vmc_lr`, `diag_shift` | `s=0` ground-state preparation |
| `q`, `snr_atol`, `rcond`, `rcond_smooth` | TDVP-Schmitt regularization (`q`>0 enables blurred sampling) |
| `integrator`, `dt_min`, `dt_max` | ODE integrator (`rk45`/`heun`/`euler`) and step limits |
| `cache_within_step` | sample+blur once per ODE step (stage 0), reweight cached samples for the remaining RK stages (blur path only) |
| `distributed_eigh` | multi-GPU QGT eigendecomposition (needs `jaxmg`) |
| `n_save_times`, `every_n_steps` | checkpoint snapshot times / interval |
| `wandb`, `data_dir`, `instance_dir`, `corr_dir`, `experiment_name` | logging backend, output root, input dirs, experiment folder |

## Outputs

`get_save_path(config)` builds a unique, human-readable directory under
`data/tvmc/<experiment>/<topology>_<shape>/instance_…/t_a_…/<model>/<sampler…>/q…/<integrator…>/<hash>/`,
ending in an 8-hex digest of the full config (any config change → new folder). It contains
`state.nk` (latest), `state_t{…}.nk` (snapshots), `log` (the logger), and the `corrs_n*_t*.npz`
written by verification. `run_tvmc.py` and `verify_tvmc.py` both call `get_save_path` so they
always agree on the location.