"""Measured CPU/MPS selection for the actual PPO policy and rollout workload.

Calibration uses disposable checkpoint copies. It never trains the real policy
or counts calibration samples as learning. Timings are estimates, not promises
about sustained performance under changing system load.
"""
from io import BytesIO
import os
import random
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure


MIN_MPS_GAIN = 0.10


def require_device(device="cpu"):
    if device not in ("mps", "cpu"):
        raise ValueError("Choose mps or cpu")
    torch.set_num_threads(1)
    if device == "mps":
        if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1":
            raise RuntimeError("Unset PYTORCH_ENABLE_MPS_FALLBACK to verify GPU operations")
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS GPU is unavailable. Use native ARM64 Python and an MPS-enabled PyTorch; "
                               "run from the macOS terminal with GPU access. No CPU fallback was selected.")
        probe = torch.ones(32, 32, device="mps", requires_grad=True)
        probe.square().mean().backward()
        torch.mps.synchronize()
        if probe.grad.device.type != "mps":
            raise RuntimeError("MPS backward verification failed")
    return {"requested_device": device, "policy_device": device, "physics_device": "cpu",
            "mps_available": torch.backends.mps.is_available(), "torch_version": torch.__version__,
            "cpu_fallback": False, "backward_probe_passed": device == "mps"}


def policy_device_report(model):
    parameters = list(model.policy.parameters())
    gradients = [p.grad for p in parameters if p.grad is not None]
    moments = [state[key] for state in model.policy.optimizer.state.values()
               for key in ("exp_avg", "exp_avg_sq") if key in state]
    result = {"parameter_devices": sorted({p.device.type for p in parameters}),
              "gradient_devices": sorted({p.device.type for p in gradients}),
              "gradient_tensors": len(gradients), "parameter_count": sum(p.numel() for p in parameters),
              "optimizer_moment_devices": sorted({p.device.type for p in moments}),
              "optimizer_moment_tensors": len(moments)}
    expected = [model.device.type]
    if (result["parameter_devices"] != expected
            or (gradients and result["gradient_devices"] != expected)
            or (moments and result["optimizer_moment_devices"] != expected)):
        raise RuntimeError("Policy, gradients or optimizer moments left the selected device")
    if model.device.type == "mps":
        torch.mps.synchronize()
        result["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
    return result


def _sync(device):
    if device == "mps":
        torch.mps.synchronize()


def _sample_physics(env):
    """Sample the actual vectorized physics/sensor/IPC cost, without learning."""
    env.seed(910001)
    obs = env.reset()
    frames, elapsed = [], []
    actions = np.zeros((env.num_envs, *env.action_space.shape), dtype=np.float32)
    for i in range(16):
        started = time.perf_counter()
        obs, _, _, _ = env.step(actions)
        seconds = time.perf_counter() - started
        if i >= 4:
            elapsed.append(seconds)
            frames.append(obs.copy())
    return np.stack(frames), float(np.median(elapsed))


def _benchmark_policy(snapshot, device, frames):
    """Time SB3 inference/transfers and real PPO.train on repeated sensory data.

Synthetic advantages are used only to exercise the actual loss/Adam kernels.
The snapshot's shapes, rollout size, batch size, epochs and optimizer are kept.
"""
    candidate = PPO.load(BytesIO(snapshot), device=device, custom_objects={"verbose": 0})
    candidate.set_logger(configure(folder=None, format_strings=[]))
    candidate.policy.set_training_mode(False)

    def inference():
        with torch.no_grad():
            actions, values, log_probs = candidate.policy(torch.as_tensor(frames[0], device=device))
        # SB3 sends all three back to CPU during rollout collection.
        return (actions.cpu().numpy(), values.clone().cpu().numpy(), log_probs.clone().cpu().numpy())

    for _ in range(8):
        inference()
    inference_times = []
    for _ in range(3):
        _sync(device)
        started = time.perf_counter()
        for _ in range(32):
            inference()
        _sync(device)
        inference_times.append((time.perf_counter() - started) / 32)

    buffer = candidate.rollout_buffer
    shape = (candidate.n_steps, candidate.n_envs)
    observations = frames[np.arange(candidate.n_steps) % len(frames)]
    rng = np.random.default_rng(910002)
    advantages = rng.normal(size=shape).astype(np.float32)

    def update():
        buffer.reset()
        buffer.observations[:] = observations
        buffer.advantages[:] = advantages
        buffer.returns[:] = advantages
        buffer.full = True
        buffer.pos = buffer.buffer_size
        candidate.train()

    update()  # Warm up kernels and Adam buffers before measuring.
    update_times = []
    for _ in range(3):
        _sync(device)
        started = time.perf_counter()
        update()
        _sync(device)
        update_times.append(time.perf_counter() - started)
    audit = policy_device_report(candidate)
    if not audit["gradient_tensors"] or not all(torch.isfinite(p).all().item() for p in candidate.policy.parameters()):
        raise RuntimeError("Calibration produced invalid policy updates")
    return {"inference_seconds_per_vector_step": float(np.median(inference_times)),
            "ppo_update_seconds_per_rollout": float(np.median(update_times)),
            "inference_samples_seconds": inference_times, "update_samples_seconds": update_times,
            "device_audit": audit}


def choose_from_timings(timings, physics_seconds, rollout_steps):
    totals = {device: rollout_steps * (physics_seconds + values["inference_seconds_per_vector_step"])
              + values["ppo_update_seconds_per_rollout"] for device, values in timings.items()}
    if not all(np.isfinite(value) and value > 0 for value in totals.values()):
        raise ValueError("Device calibration timings must be finite and positive")
    gain = 1.0 - totals["mps"] / totals["cpu"]
    selected = "mps" if gain >= MIN_MPS_GAIN else "cpu"
    return selected, {"estimated_rollout_seconds": totals, "estimated_mps_gain_fraction": gain,
                      "minimum_mps_gain_fraction": MIN_MPS_GAIN}


def select_training_device(model, env, requested="auto"):
    """Choose once per launch/resume; changing a live optimizer mid-rollout is avoided."""
    if requested not in ("auto", "cpu", "mps"):
        raise ValueError("Choose auto, cpu or mps")
    if requested != "auto":
        return {**require_device(requested), "selection_reason": "explicit_override"}
    result = {**require_device("cpu"), "requested_device": "auto", "benchmark": None}
    try:
        mps_info = require_device("mps")
    except RuntimeError as error:
        return {**result, "selection_reason": "mps_unavailable_or_unverifiable", "mps_error": str(error)}
    result["backward_probe_passed"] = mps_info["backward_probe_passed"]
    print("Calibrating CPU and MPS using this policy, environment count and PPO workload...", flush=True)
    # PPO.load initializes policies and sets seeds. Preserve caller RNG streams.
    rng_state = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.mps.get_rng_state())
    try:
        frames, physics_seconds = _sample_physics(env)
        stream = BytesIO()
        model.save(stream)
        snapshot = stream.getvalue()
        timings = {"cpu": _benchmark_policy(snapshot, "cpu", frames)}
        try:
            timings["mps"] = _benchmark_policy(snapshot, "mps", frames)
        except (RuntimeError, NotImplementedError) as error:
            return {**result, "selection_reason": "mps_policy_benchmark_failed", "mps_error": str(error),
                    "benchmark": {"timings": timings, "physics_seconds_per_vector_step": physics_seconds}}
        selected, comparison = choose_from_timings(timings, physics_seconds, model.n_steps)
        return {**(mps_info if selected == "mps" else result), "requested_device": "auto",
                "policy_device": selected, "selection_reason": ("mps_measured_advantage" if selected == "mps"
                                                                    else "cpu_preferred_for_measured_workload"),
                "benchmark": {"method": "sampled_physics_plus_policy_and_ppo_update", "timings": timings,
                              "physics_seconds_per_vector_step": physics_seconds,
                              "envs": env.num_envs, "rollout_steps": model.n_steps,
                              "batch_size": model.batch_size, "epochs": model.n_epochs,
                              "observation_shape": list(model.observation_space.shape), **comparison}}
    finally:
        random.setstate(rng_state[0])
        np.random.set_state(rng_state[1])
        torch.set_rng_state(rng_state[2])
        torch.mps.set_rng_state(rng_state[3])
