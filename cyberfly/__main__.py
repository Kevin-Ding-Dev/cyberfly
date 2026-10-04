"""Run `python -m cyberfly --help` from the project directory."""
import argparse
import json
from pathlib import Path
import time

import numpy as np

from .env import CyberflyEnv, TaskConfig


def rollout(args):
    model = None
    if args.model:
        import torch
        from stable_baselines3 import PPO
        from .learning import model_config
        torch.set_num_threads(1)
        model = PPO.load(args.model, device="cpu")
        config = model_config(model)
    else:
        config = TaskConfig(episode_seconds=args.seconds, heading_range=0.0)
    if args.viewer and args.video:
        raise ValueError("Choose either --viewer or --video")
    env = CyberflyEnv(config, render_mode="rgb_array" if args.video else None)
    viewer = None
    writer = None
    from .learning import write_json
    try:
        obs, _ = env.reset(seed=args.seed)
        if args.viewer:
            import mujoco.viewer
            viewer = mujoco.viewer.launch_passive(env.sim.mj_model, env.sim.mj_data)
            with viewer.lock():
                viewer.cam.distance = 12
                viewer.cam.azimuth = 135
                viewer.cam.elevation = -25
                viewer.cam.lookat[:] = env.position
        if args.video:
            import imageio.v2 as imageio
            Path(args.video).parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(args.video, fps=25, codec="libx264",
                                        macro_block_size=16)
        stride = max(1, round(1 / (25 * config.control_dt)))
        count = 0
        started = time.perf_counter()
        info = {}
        while viewer is None or viewer.is_running():
            tick = time.perf_counter()
            action = (np.zeros(2, dtype=np.float32) if model is None
                      else model.predict(obs, deterministic=True)[0])
            obs, _, terminated, truncated, info = env.step(action)
            count += 1
            if writer is not None and count % stride == 0:
                writer.append_data(env.render())
            if viewer is not None:
                with viewer.lock():
                    viewer.cam.lookat[:] = env.position
                viewer.sync()
                time.sleep(max(0, config.control_dt - (time.perf_counter() - tick)))
            if terminated or truncated:
                break
        result = {
            "policy": "neural policy" if model is not None else "fixed CPG baseline",
            "training_timesteps": int(model.num_timesteps) if model is not None else None,
            "task": config.to_dict(), "seed": args.seed,
            "completed": "metrics" in info,
            "wall_seconds": time.perf_counter() - started,
            "metrics": info.get("metrics"),
        }
        write_json(args.output, result)
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
        if args.video:
            print(f"Video: {Path(args.video).resolve()}", flush=True)
    finally:
        if viewer is not None:
            viewer.close()
        if writer is not None:
            writer.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description="Cyberfly: baseline, PPO training, evaluation")
    subs = parser.add_subparsers(dest="command", required=True)
    from .forage_cli import add_commands
    add_commands(subs)
    demo = subs.add_parser("walk", help="Run the baseline or a trained checkpoint")
    demo.add_argument("--seconds", type=float, default=2.0,
                      help="Baseline duration; learned policies use their saved task duration")
    demo.add_argument("--seed", type=int, default=0)
    demo.add_argument("--viewer", action="store_true")
    demo.add_argument("--video")
    demo.add_argument("--model")
    demo.add_argument("--output", default="outputs/walk.json")
    training = subs.add_parser("train", help="Train a bounded number of additional env steps")
    training.add_argument("--run", required=True)
    training.add_argument("--steps", type=int, default=2048)
    training.add_argument("--resume")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--task", choices=["straight", "turn"])
    training.add_argument("--episode-seconds", type=float)
    training.add_argument("--rollout-steps", type=int, default=256)
    training.add_argument("--eval-every", type=int, default=1024)
    training.add_argument("--eval-episodes", type=int, default=3)
    testing = subs.add_parser("evaluate", help="Compare checkpoints on independent test seeds")
    testing.add_argument("--model", required=True)
    testing.add_argument("--reference")
    testing.add_argument("--episodes", type=int, default=10)
    testing.add_argument("--seed-start", type=int, default=20000)
    testing.add_argument("--output", default="outputs/evaluation.json")
    subs.add_parser("check", help="Run environment contract and reproducibility checks")
    args = parser.parse_args()
    if args.command.startswith("forage"):
        from .forage_cli import dispatch
        dispatch(args)
    elif args.command == "walk":
        rollout(args)
    elif args.command == "train":
        from .learning import train
        train(args)
    elif args.command == "evaluate":
        from .learning import compare
        compare(args)
    else:
        from .checks import run_checks
        run_checks()


if __name__ == "__main__":
    main()
