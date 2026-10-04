"""Bounded PPO runs, validation checkpoints, and independent test episodes."""
import json
import platform
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .env import CyberflyEnv, TaskConfig


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False)
                    + "\n", encoding="utf-8")
    temp.replace(path)


def save_model(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.stem + ".tmp.zip")
    model.save(temp)
    temp.replace(path)


def model_config(model):
    value = getattr(model, "cyberfly_config", None)
    if value is None:
        raise ValueError("This is not a Cyberfly checkpoint")
    config = TaskConfig(**value)
    if config.version != TaskConfig().version:
        raise ValueError("Checkpoint task version differs from current code")
    return config


def evaluate(env, model, seeds):
    """Each policy sees the same command and initialization for each seed."""
    rows = []
    for seed in seeds:
        obs, _ = env.reset(seed=seed)
        while True:
            action = (np.zeros(2, dtype=np.float32) if model is None
                      else model.predict(obs, deterministic=True)[0])
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                rows.append({"seed": seed, **info["metrics"]})
                break
    return {
        "episodes": len(rows), "seeds": list(seeds),
        "mean_return": float(np.mean([r["return"] for r in rows])),
        "std_return": float(np.std([r["return"] for r in rows])),
        "mean_velocity_error_mm_s": float(np.mean([
            r["mean_velocity_error_mm_s"] for r in rows])),
        "mean_heading_error_deg": float(np.mean([
            r["mean_heading_error_deg"] for r in rows])),
        "fall_rate": float(np.mean([r["fallen"] for r in rows])),
        "mean_distance_mm": float(np.mean([r["distance_mm"] for r in rows])),
        "rows": rows,
    }


class ValidationCallback(BaseCallback):
    """Select by a fixed validation set, never by training episode returns."""
    def __init__(self, run_dir, config, every, episodes):
        super().__init__()
        self.run_dir = Path(run_dir)
        self.env = CyberflyEnv(config)
        self.every = every
        self.seeds = list(range(10_000, 10_000 + episodes))
        self.best = -np.inf
        self.last_eval = -1

    def assess(self):
        stats = evaluate(self.env, self.model, self.seeds)
        stats["timesteps"] = self.model.num_timesteps
        with (self.run_dir / "validation.jsonl").open("a") as f:
            f.write(json.dumps(stats, allow_nan=False) + "\n")
        for key in ("mean_return", "mean_velocity_error_mm_s", "fall_rate"):
            self.logger.record("validation/" + key, stats[key])
        self.logger.dump(self.model.num_timesteps)
        save_model(self.model, self.run_dir / "checkpoints" /
                   f"step_{self.model.num_timesteps:09d}.zip")
        save_model(self.model, self.run_dir / "latest.zip")
        if stats["mean_return"] > self.best:
            self.best = stats["mean_return"]
            save_model(self.model, self.run_dir / "best.zip")
            write_json(self.run_dir / "best_validation.json", stats)
        self.last_eval = self.model.num_timesteps
        print(f"VALIDATION steps={self.last_eval} return={stats['mean_return']:.3f} "
              f"velocity_error={stats['mean_velocity_error_mm_s']:.3f} mm/s "
              f"fall_rate={stats['fall_rate']:.0%}", flush=True)

    def _on_training_start(self):
        self.assess()

    def _on_rollout_start(self):
        # PPO has finished the previous update when a new rollout starts.
        if self.model.num_timesteps - self.last_eval >= self.every:
            self.assess()

    def _on_step(self):
        return True

    def _on_training_end(self):
        if self.model.num_timesteps != self.last_eval:
            self.assess()


def train(args):
    run_dir = Path(args.run).resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise ValueError("Output directory is not empty. Use a new --run directory; "
                         "--resume loads an old checkpoint into a NEW run.")
    if args.steps <= 0 or args.eval_every <= 0 or args.eval_episodes <= 0:
        raise ValueError("Training and evaluation counts must be positive")
    if args.rollout_steps < 64 or args.rollout_steps % 64:
        raise ValueError("--rollout-steps must be a multiple of 64")
    torch.set_num_threads(1)
    model = None
    if args.resume:
        model = PPO.load(args.resume, device="cpu")
        config = model_config(model)
        if args.episode_seconds is not None and args.episode_seconds != config.episode_seconds:
            raise ValueError("Resume must keep the original episode duration")
        if args.task is not None:
            expected = 0.0 if args.task == "straight" else 0.35
            if config.heading_range != expected:
                raise ValueError("Resume must keep the original task")
    else:
        config = TaskConfig(episode_seconds=args.episode_seconds or 2.0,
                            heading_range=0.35 if args.task == "turn" else 0.0)
    run_dir.mkdir(parents=True, exist_ok=True)
    env = Monitor(CyberflyEnv(config), str(run_dir / "monitor.csv"))
    callback = None
    started = time.perf_counter()
    interrupted = False
    try:
        if model is None:
            model = PPO(
                "MlpPolicy", env, device="cpu", seed=args.seed, verbose=1,
                n_steps=args.rollout_steps, batch_size=64, n_epochs=5,
                learning_rate=3e-4, gamma=0.99, gae_lambda=0.95,
                ent_coef=0.005, policy_kwargs={"net_arch": [64, 64]},
                tensorboard_log=str(run_dir / "tb"),
            )
            model.cyberfly_config = config.to_dict()
        else:
            model.set_env(env)
            model.tensorboard_log = str(run_dir / "tb")
        initial_steps = model.num_timesteps
        write_json(run_dir / "config.json", {
            "task": config.to_dict(), "arguments": vars(args),
            "initial_timesteps": initial_steps, "architecture": platform.machine(),
            "packages": {name: version(name) for name in (
                "flygym", "mujoco", "torch", "gymnasium", "stable-baselines3")},
        })
        save_model(model, run_dir / "initial.zip")
        callback = ValidationCallback(run_dir, config, args.eval_every, args.eval_episodes)
        print("Starting bounded training; Ctrl+C saves latest.zip. "
              "The first validation runs before learning.", flush=True)
        try:
            model.learn(total_timesteps=args.steps, callback=callback,
                        reset_num_timesteps=False, tb_log_name="ppo")
        except KeyboardInterrupt:
            interrupted = True
            print("Interrupted: saving current policy and optimizer.", flush=True)
        finally:
            # Includes optimizer and counter, but not an in-progress rollout or
            # the exact physics/RNG state. Resuming starts fresh episodes.
            save_model(model, run_dir / "latest.zip")
        write_json(run_dir / "summary.json", {
            "initial_timesteps": initial_steps, "final_timesteps": model.num_timesteps,
            "added_timesteps": model.num_timesteps - initial_steps,
            "wall_seconds": time.perf_counter() - started,
            "interrupted": interrupted,
        })
        print(f"Saved {run_dir / 'latest.zip'}", flush=True)
    finally:
        env.close()
        if callback is not None:
            callback.env.close()


def compare(args):
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    torch.set_num_threads(1)
    model = PPO.load(args.model, device="cpu")
    config = model_config(model)
    env = CyberflyEnv(config)
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    try:
        results = {}
        policies = [("baseline", None), ("trained", model)]
        if args.reference:
            reference = PPO.load(args.reference, device="cpu")
            if model_config(reference) != config:
                raise ValueError("Reference and trained task configurations differ")
            policies.insert(1, ("reference", reference))
        for name, policy in policies:
            print(f"Evaluating {name}: {args.episodes} episodes...", flush=True)
            results[name] = evaluate(env, policy, seeds)
            stats = results[name]
            print(f"{name}: return={stats['mean_return']:.3f} "
                  f"velocity_error={stats['mean_velocity_error_mm_s']:.3f} mm/s "
                  f"heading_error={stats['mean_heading_error_deg']:.2f} deg "
                  f"falls={stats['fall_rate']:.0%}", flush=True)
        reference_stats = results.get("reference", results["baseline"])
        current = results["trained"]
        results["comparison"] = {
            "against": "reference" if "reference" in results else "baseline",
            "return_delta": current["mean_return"] - reference_stats["mean_return"],
            "velocity_error_delta_mm_s": current["mean_velocity_error_mm_s"]
                - reference_stats["mean_velocity_error_mm_s"],
            "fall_rate_delta": current["fall_rate"] - reference_stats["fall_rate"],
            "note": "Finite test set; no guarantee of generalization or monotonic improvement.",
        }
        write_json(args.output, {"task": config.to_dict(), "model": str(args.model),
                                 "model_timesteps": int(model.num_timesteps),
                                 **results})
        print(f"Report: {Path(args.output).resolve()}", flush=True)
    finally:
        env.close()
