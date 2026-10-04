"""Maze demonstrations, workload-aware training and held-out evaluation."""
import csv
import json
from pathlib import Path
import time

import numpy as np

from .learning import write_json
from .maze import MazeConfig
from .maze_env import MazeEnv


def add_commands(subs):
    demo = subs.add_parser("maze",help="Single-source odor-guided physical maze foraging")
    demo.add_argument("--config")
    demo.add_argument("--model",help="Maze checkpoint or run directory (best validated checkpoint)")
    demo.add_argument("--seed",type=int,default=0)
    demo.add_argument("--viewer",action="store_true")
    demo.add_argument("--video")
    demo.add_argument("--episodes",type=int,default=1,help="Sequential seeds; 0 loops until stopped")
    demo.add_argument("--output",default="outputs/maze-demo")
    train = subs.add_parser("maze-train",help="Cumulative PPO training; benchmark CPU/MPS by default")
    train.add_argument("--run",required=True)
    train.add_argument("--config")
    train.add_argument("--steps",type=int,default=16384,help="Additional steps; 0 trains until Ctrl+C")
    train.add_argument("--envs",type=int,default=4)
    train.add_argument("--rollout-steps",type=int,default=256)
    train.add_argument("--eval-every",type=int,default=8192)
    train.add_argument("--eval-episodes",type=int,default=6)
    train.add_argument("--device",choices=("auto","mps","cpu"),default="auto",
                       help="auto benchmarks the real workload; cpu/mps force a device")
    train.add_argument("--seed",type=int,default=42)
    test = subs.add_parser("maze-evaluate",help="Evaluate a frozen maze policy on independent seeds")
    test.add_argument("--model",required=True)
    test.add_argument("--episodes",type=int,default=12)
    test.add_argument("--seed-start",type=int,default=300000)
    test.add_argument("--device",choices=("mps","cpu"),default="cpu",
                      help="Single-environment inference defaults to CPU")
    test.add_argument("--output",default="outputs/maze-evaluation.json")
    device = subs.add_parser("maze-device",help="Verify Apple GPU forward and backward operations")
    device.add_argument("--device",choices=("mps","cpu"),default="mps")
    subs.add_parser("maze-check",help="Check connectivity, odor, vision, rewards and physical ingestion")


def rollout_maze(args):
    from .maze_learning import load_maze_policy
    import torch
    torch.set_num_threads(1)
    if args.viewer and args.video or args.episodes < 0 or args.seed < 0:
        raise ValueError("Choose viewer or video, nonnegative seed and episodes")
    if args.video and args.episodes != 1:
        raise ValueError("Record one episode per --video; use --episodes with the viewer")
    if args.model:
        if args.config:
            raise ValueError("A maze checkpoint uses its saved configuration")
        model,config,provenance = load_maze_policy(args.model)
    else:
        model,provenance = None,None
        config = MazeConfig(**json.loads(Path(args.config).read_text())) if args.config else MazeConfig()
    root = Path(args.output)
    root.mkdir(parents=True,exist_ok=True)
    env = MazeEnv(config,"rgb_array" if args.video else None)
    viewer = writer = None
    episode = 0
    try:
        while args.episodes == 0 or episode < args.episodes:
            seed = args.seed+episode
            obs,initial = env.reset(seed=seed)
            if args.viewer and viewer is None:
                import mujoco.viewer
                viewer = mujoco.viewer.launch_passive(env.sim.mj_model,env.sim.mj_data)
                with viewer.lock():
                    viewer.cam.distance = env.camera.distance
                    viewer.cam.azimuth = env.camera.azimuth
                    viewer.cam.elevation = env.camera.elevation
                    viewer.cam.lookat[:] = env.camera.lookat
            if args.video:
                import imageio.v2 as imageio
                Path(args.video).parent.mkdir(parents=True,exist_ok=True)
                writer = imageio.get_writer(args.video,fps=25,codec="libx264",macro_block_size=16)
            path = root if args.episodes == 1 else root/f"episode_{episode+1:05d}_seed_{seed}"
            path.mkdir(parents=True,exist_ok=True)
            write_json(path/"maze.json",initial["maze"])
            print(f"MAZE seed={seed}, kind={env.layout.kind}, target depth={env.layout.depth[env.layout.goal]}, sugar sources=1",flush=True)
            info,frames = {},[]
            for step in range(env.max_steps):
                if viewer is not None and not viewer.is_running():
                    break
                tick = time.perf_counter()
                action = np.zeros(4) if model is None else model.predict(obs,deterministic=True)[0]
                obs,_,terminated,truncated,info = env.step(action)
                if writer is not None and ((step+1)%2 == 0 or terminated or truncated):
                    writer.append_data(env.render())
                    frames.append(info["time_s"])
                if viewer is not None:
                    viewer.sync()
                    time.sleep(max(0,env.config.control_dt-(time.perf_counter()-tick)))
                if (step+1)%100 == 0:
                    print(f"t={info['time_s']:.1f}s mode={info['mode']} odor={info['odor']:.3f} "
                          f"intake={info['consumed']:.2f}/1 dopamine={info['dopamine_appetitive']:.3f}",flush=True)
                if terminated or truncated:
                    break
            if writer is not None:
                writer.close()
                writer = None
            with (path/"trajectory.csv").open("w",newline="") as handle:
                table = csv.DictWriter(handle,fieldnames=list(env.trajectory[0]))
                table.writeheader()
                table.writerows(env.trajectory)
            write_json(path/"events.json",env.events)
            result = {"task":"maze","config":config.to_dict(),"seed":seed,"maze":initial["maze"],
                      "model_provenance":provenance,"training_timesteps":int(model.num_timesteps) if model else 0,
                      "complete":"metrics" in info,"metrics":info.get("metrics"),
                      "video":({"fps":25,"frames":len(frames),"duration_s":len(frames)/25,
                                "duration_error_s":len(frames)/25-info["time_s"],"frame_times_s":frames} if args.video else None),
                      "camera":{"distance":env.camera.distance,"azimuth":env.camera.azimuth,
                                "elevation":env.camera.elevation,"lookat":env.camera.lookat.tolist(),"follow_body":False},
                      "biological_similarity":None}
            write_json(path/"summary.json",result)
            print(json.dumps({"seed":seed,"complete":result["complete"],"metrics":result["metrics"]},ensure_ascii=False),flush=True)
            episode += 1
            if viewer is not None and not viewer.is_running():
                break
    finally:
        if writer is not None:
            writer.close()
        if viewer is not None:
            viewer.close()
        env.close()


def dispatch(args):
    if args.command == "maze":
        rollout_maze(args)
    elif args.command == "maze-train":
        from .maze_learning import train_maze
        train_maze(args)
    elif args.command == "maze-device":
        from .maze_learning import require_device
        print(json.dumps(require_device(args.device),indent=2))
    elif args.command == "maze-evaluate":
        from .maze_learning import require_device,load_maze_policy,evaluate_maze
        if args.episodes < 1 or args.seed_start < 0:
            raise ValueError("Use positive episodes and nonnegative seeds")
        device = require_device(args.device)
        model,config,provenance = load_maze_policy(args.model,args.device)
        env = MazeEnv(config)
        try:
            stats = evaluate_maze(env,model,list(range(args.seed_start,args.seed_start+args.episodes)))
            write_json(args.output,{"model_provenance":provenance,"device":device,"evaluation":stats})
            print(json.dumps({key:value for key,value in stats.items() if key != "rows"},indent=2))
        finally:
            env.close()
    elif args.command == "maze-check":
        from .maze_checks import run_checks
        run_checks()
