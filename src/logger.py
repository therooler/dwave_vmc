import netket as nk
from netket.utils import struct
from netket.utils.types import ArrayLike

from netket.callbacks import AbstractCallback
from nqxpack import load as nqx_load
from nqxpack import save as nqx_save

import jax
import jax.numpy as jnp

import numpy as np


class CheckpointCallback(AbstractCallback):
    """Periodically saves a resumable snapshot (state + log).

    Fires at ``on_step_end`` (after ``update_parameters``), so the saved
    parameters and time are post-update.
    """

    output_dir: str = struct.field(pytree_node=False)
    logger: object = struct.field(pytree_node=False)
    every_n_steps: int = struct.field(pytree_node=False)
    save_times: ArrayLike = struct.field(pytree_node=False)
    save_times_tracked: ArrayLike = struct.field(pytree_node=False)
    done: bool = struct.field(pytree_node=False)
    verbose: bool = struct.field(pytree_node=False)
    wandb_run: None

    def __init__(
        self, output_dir, logger, every_n_steps, wandb_run=None, save_times=None, verbose=False
    ):
        self.output_dir = output_dir
        self.logger = logger
        self.every_n_steps = every_n_steps
        if wandb_run is not None:
            from wandb.sdk.wandb_run import Run

            if not isinstance(wandb_run, Run):
                raise ValueError(
                    f"`wandb_run` must be None or the object returned by wandb.init(), received: {wandb_run}"
                )

        self.wandb_run = wandb_run
        self.done = False
        self.save_times = np.array([]) if save_times is None else save_times
        self.verbose = verbose

    def on_step_end(self, step, log_data, driver):
        hit = np.isclose(step, self.save_times, atol=driver.dt)
        if np.any(hit):
            candidates = np.where(hit)[0]
            idx = int(candidates[np.argmin(np.abs(self.save_times[candidates] - step))])
            nqx_save(
                driver.state,
                self.output_dir + f"/state_t{float(self.save_times[idx]):1.3f}.nk",
            )
            self.save_times[idx] = -1
        if driver._step_count % self.every_n_steps != 0:
            return
        # Save metadata (t, step) so restore doesn't depend on dt
        if jax.process_index() == 0:
            self.logger.serialize(self.output_dir + "/log")
            nqx_save(driver.state, self.output_dir + f"/state.nk")
            if self.wandb_run is not None:
                data = self.logger.data.to_dict()
                if data:
                    self.wandb_run.log(_clean(data))
            if self.verbose:
                print(f"  Checkpoint at step {driver._step_count}")

    def __call__(self, step, log_data, driver):
        self.on_step_end(step, log_data, driver)
        return True

    def restore_state(self, name: str | None = None):
        try:
            fname = f"/state.nk" if name is None else name
            vs = nqx_load(self.output_dir + fname)
        except FileNotFoundError:
            print(f"No checkpoint found in {self.output_dir}")
            return None

        return vs

    def restore_logger(self):
        try:
            self.logger = nk.logging.RuntimeLog.deserialize(self.output_dir + f"/log")
            try:
                self.done = self.logger["done"]
            except KeyError:
                self.done = False
        except FileNotFoundError:
            print(f"No logger found in {self.output_dir}")
            return self.logger, False
        # step = logger.data["step"][-1]
        return self.logger, True

    def finish(self, state):
        self.logger.data["done"] = nk.utils.History(True)
        if jax.process_index() == 0:
            self.logger.serialize(self.output_dir + "/log")
            nqx_save(state, self.output_dir + f"/state.nk")
            if self.wandb_run is not None:
                data = self.logger.data.to_dict()
                if data:
                    self.wandb_run.log(_clean(data))


def _clean(data):
    # wandb gets scalars only. Array-valued histories (per-step vectors like
    # `pt_accept_per_beta`, `0/snr`, `0/ev`, `0/ev_reg`) are skipped here; they
    # remain in full in `log.json` via `logger.serialize`.
    new_data = {}
    for k_main, history in data.items():
        for k, v in history.to_dict().items():
            if k == "iters":
                continue
            if np.iscomplexobj(v[0]):
                if np.ndim(v.real[-1]) > 0:
                    continue
                new_data[f"{k_main}.{k}Re"] = v.real[-1]
                new_data[f"{k_main}.{k}Im"] = v.imag[-1]
            else:
                if np.ndim(v[-1]) > 0:
                    continue
                new_data[f"{k_main}.{k}"] = v[-1]
    return new_data
