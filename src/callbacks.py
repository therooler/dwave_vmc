import numpy as np



def get_umbrella_monitor_callback(save_times, save_path):
    """Callback that logs the relevant data returned by TDVPBlurred"""
    save_times_tracked = save_times.copy()

    def umbrella_monitor_callback(step, log, driver):
        # Populate monitoring metrics from driver's self._monitor (make_monitor_dict)
        try:
            dt = driver.integrator._state.dt
            log["dt"] = dt
        except AttributeError:
            log["dt"] = np.nan
        try:
            monitor = driver._monitor
        except AttributeError:
            raise ValueError("No monitor found in driver, callback can't be used.")

        # Scalars: convert possible JAX arrays to Python floats
        def _to_float(x, default=np.nan):
            try:
                return float(np.array(x))
            except Exception:
                return default

        log["r_squared"] = _to_float(monitor.get("rmd", np.nan))
        # ESS as fraction in [0,1] for plotting, plus absolute ESS
        log["ess_bridge"] = _to_float(monitor.get("ess_bridge", np.nan))

        log["snr_min"] = _to_float(monitor.get("snr_min", np.nan))
        log["snr_10p"] = _to_float(monitor.get("snr_10p", np.nan))
        log["snr_med"] = _to_float(monitor.get("snr_med", np.nan))
        log["snrF_min"] = _to_float(monitor.get("snrF_min", np.nan))
        log["snrF_med"] = _to_float(monitor.get("snrF_med", np.nan))
        # Current bridge parameter q (kept in [0,1])
        try:
            log["q_bridge"] = _to_float(driver.q, np.nan)
        except AttributeError:
            # Randomized bridge has two values of q
            try:
                log["q_bridge"] = (_to_float(driver.q1, np.nan), _to_float(driver.q2, np.nan))
            except AttributeError:
                log["q_bridge"] = np.nan
                
        hit = np.isclose(step, save_times_tracked, atol=driver.dt)
        if np.any(hit):
            idx = np.where(np.isclose(step, save_times_tracked, atol=driver.dt))[0]
            save_times_tracked[idx] = -1
            log["snr"] = monitor.get("snr", np.nan)
            log["snr_F"] = monitor.get("snr_F", np.nan)
            ev = monitor.get("ev", np.array([np.nan]))
            ev_reg = monitor.get("ev_reg", np.array([np.nan]))
            np.save(f"{save_path}/ev_{idx[0]}.npy", ev)
            np.save(f"{save_path}/ev_reg_{idx[0]}.npy", ev_reg)

        return True

    return umbrella_monitor_callback