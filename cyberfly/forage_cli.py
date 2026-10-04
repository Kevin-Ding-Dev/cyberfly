"""CLI for random sugar foraging, videos, training loops and biology checks."""
import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np

from .foraging import FOOD_PROFILES, ForageConfig, ForagingEnv, apply_food_profile
from .learning import write_json


def add_commands(subs):
    demo=subs.add_parser("forage",help="Search for random sugar patches, stop and ingest")
    demo.add_argument("--config")
    demo.add_argument("--seconds",type=float)
    demo.add_argument("--seed",type=int,default=0)
    demo.add_argument("--food-count",type=int,choices=range(1,13),
                      help="Test with exactly N sugar patches; does not change the checkpoint")
    demo.add_argument("--food-profile",choices=FOOD_PROFILES,help="Food geometry preset for this test")
    demo.add_argument("--model",help="Foraging .zip checkpoint or training-loop directory")
    demo.add_argument("--viewer",action="store_true")
    demo.add_argument("--camera-distance",type=float,help="Fixed camera distance; larger shows more of the arena")
    demo.add_argument("--video")
    demo.add_argument("--output",default="outputs/forage")
    batch=subs.add_parser("forage-batch",help="Record a count-by-seed matrix with one frozen model")
    batch.add_argument("--model",required=True,help="Foraging checkpoint or training-loop directory")
    batch.add_argument("--food-counts",type=int,nargs="+",choices=range(1,13),default=[2,5,8])
    batch.add_argument("--food-profile",choices=FOOD_PROFILES,help="Food geometry preset for every video")
    batch.add_argument("--seeds",type=int,nargs="+",default=[12,13,14])
    batch.add_argument("--camera-distance",type=float,help="One fixed distance for every video (default: 30)")
    batch.add_argument("--output",required=True,help="New or empty directory for videos and reports")
    loop=subs.add_parser("forage-loop",help="Train challengers and validate before promotion")
    loop.add_argument("--run",required=True)
    loop.add_argument("--config")
    loop.add_argument("--rounds",type=int,default=3,help="Additional rounds; 0 runs until Ctrl+C")
    loop.add_argument("--steps",type=int,default=2048)
    loop.add_argument("--seed",type=int,default=42)
    loop.add_argument("--eval-episodes",type=int,default=8)
    loop.add_argument("--reference",help="Matched real-trajectory reference manifest")
    evaluate=subs.add_parser("forage-evaluate",help="Independent test of a foraging checkpoint")
    evaluate.add_argument("--model",required=True)
    evaluate.add_argument("--episodes",type=int,default=10)
    evaluate.add_argument("--seed-start",type=int,default=300000)
    evaluate.add_argument("--reference")
    evaluate.add_argument("--output",default="outputs/forage-evaluation.json")
    subs.add_parser("forage-check",help="Test perception, ingestion and promotion rules")


def rollout_forage(args, *, loaded_policy=None, show_summary=True):
    from .forage_learning import load_policy_snapshot
    if args.viewer and args.video:
        raise ValueError("Use either --viewer or --video")
    camera_distance=getattr(args,"camera_distance",None)
    if camera_distance is not None and (not np.isfinite(camera_distance) or camera_distance<=0):
        raise ValueError("camera-distance must be finite and positive")
    if args.model:
        model,config,provenance=(loaded_policy if loaded_policy is not None
                                 else load_policy_snapshot(args.model))
        checkpoint=provenance["path"]
        training_config=config.to_dict()
        if args.config or args.seconds is not None:
            raise ValueError("A learned policy uses its saved task configuration")
    else:
        model=None
        checkpoint=None
        provenance=None
        training_config=None
        data=json.loads(Path(args.config).read_text()) if args.config else {}
        if args.seconds is not None:
            data["episode_seconds"]=args.seconds
        config=ForageConfig(**data)
    food_profile=getattr(args,"food_profile",None)
    config=apply_food_profile(config,food_profile)
    food_count=getattr(args,"food_count",None)
    if food_count is not None:
        config=replace(config,food_min=food_count,food_max=food_count)
    outside_range=bool(training_config is not None and food_count is not None
                       and not training_config["food_min"]<=food_count<=training_config["food_max"])
    if outside_range:
        print(f"TEST: {food_count} food patches is outside the saved training count range "
              f"{training_config['food_min']}–{training_config['food_max']}.",flush=True)
    geometry_changed=bool(training_config is not None and any(
        training_config[key]!=config.to_dict()[key]
        for key in ("food_profile","food_radius","droplet_volume_ul")))
    if geometry_changed:
        print(f"TEST: food geometry changed from training: {config.food_profile}, "
              f"diameter={2*config.food_radius:.3f} mm.",flush=True)
    overrides={}
    if food_count is not None:
        overrides["food_count"]=food_count
    if food_profile is not None:
        overrides["food_profile"]=food_profile
    path=Path(args.output)
    path.mkdir(parents=True,exist_ok=True)
    env=ForagingEnv(config,render_mode="rgb_array" if args.video else None)
    if camera_distance is not None:
        env.camera.distance=camera_distance
    viewer=None
    writer=None
    frame_times=[]
    video_fps=env.metadata["render_fps"]
    frame_stride=round(1/(video_fps*env.config.control_dt))
    try:
        obs,_=env.reset(seed=args.seed)
        layout=env.food_positions[:env.food_count].tolist()
        if args.viewer:
            import mujoco.viewer
            viewer=mujoco.viewer.launch_passive(env.sim.mj_model,env.sim.mj_data)
            with viewer.lock():
                viewer.cam.distance=env.camera.distance
                viewer.cam.azimuth=env.camera.azimuth
                viewer.cam.elevation=env.camera.elevation
                viewer.cam.lookat[:]=env.camera.lookat
        if args.video:
            import imageio.v2 as imageio
            Path(args.video).parent.mkdir(parents=True,exist_ok=True)
            writer=imageio.get_writer(args.video,fps=video_fps,codec="libx264",macro_block_size=16)
        started=time.perf_counter()
        info={}
        for step in range(env.max_steps):
            if viewer is not None and not viewer.is_running():
                break
            tick=time.perf_counter()
            action=(np.zeros(3,dtype=np.float32) if model is None
                    else model.predict(obs,deterministic=True)[0])
            obs,_,terminated,truncated,info=env.step(action)
            if writer is not None and ((step+1)%frame_stride==0 or terminated or truncated):
                writer.append_data(env.render())
                frame_times.append(float(info["time_s"]))
            if viewer is not None:
                viewer.sync()
                time.sleep(max(0,env.config.control_dt-(time.perf_counter()-tick)))
            if (step+1)%50==0:
                print(f"t={info['time_s']:.1f}s mode={info['mode']} "
                      f"consumed={info['consumed']:.2f}/{info['food_count']} "
                      f"energy={float(info['energy']):.3f}",flush=True)
            if terminated or truncated:
                break
        if env.trajectory:
            with (path/"trajectory.csv").open("w",newline="") as f:
                fields=["episode_id",*env.trajectory[0]]
                writer_csv=csv.DictWriter(f,fieldnames=fields)
                writer_csv.writeheader()
                writer_csv.writerows({"episode_id":args.seed,**row} for row in env.trajectory)
        write_json(path/"events.json",env.events)
        video=None
        if writer is not None:
            # Finalize the file before reporting a completed recording.
            writer.close()
            writer=None
            duration=float(info["time_s"])
            video={"fps":video_fps,"frames":len(frame_times),
                   "duration_s":len(frame_times)/video_fps,
                   "simulation_duration_s":duration,
                   "duration_error_s":len(frame_times)/video_fps-duration,
                   "frame_times_s":frame_times}
        result={"config":config.to_dict(),"seed":args.seed,"food_layout_mm":layout,
                "trajectory_schema_version":2,
                "model_checkpoint":str(checkpoint) if checkpoint else None,
                "model_provenance":provenance,
                "video":video,
                "training_config":training_config,
                "test_overrides":overrides,
                "food_geometry":config.food_geometry(),
                "food_geometry_differs_from_training":geometry_changed,
                "outside_training_food_count_range":outside_range,
                "default_camera":{"distance":float(env.camera.distance),
                                  "azimuth":float(env.camera.azimuth),"elevation":float(env.camera.elevation),
                                  "lookat":env.camera.lookat.tolist(),"follow_body":env.camera_follow_body},
                "controller":"sensory baseline + GRU/PPO residual" if model else "sensory reflex baseline",
                "training_timesteps":int(model.num_timesteps) if model else 0,
                "complete":"metrics" in info,"metrics":info.get("metrics"),
                "wall_seconds":time.perf_counter()-started,
                "biological_similarity":None,
                "modeling_status":"Functional ground-foraging model; no complete brain or flight simulation"}
        write_json(path/"summary.json",result)
        if show_summary:
            print(json.dumps(result,indent=2,ensure_ascii=False),flush=True)
        return result
    finally:
        if viewer is not None:
            viewer.close()
        if writer is not None:
            writer.close()
        env.close()


def record_batch(args):
    import torch
    from .forage_learning import load_policy_snapshot
    from .learning import save_model

    if any(seed<0 for seed in args.seeds):
        raise ValueError("Seeds must be nonnegative")
    if len(set(args.seeds))!=len(args.seeds) or len(set(args.food_counts))!=len(args.food_counts):
        raise ValueError("Use distinct seeds and food counts to keep every video filename unique")
    camera_distance=getattr(args,"camera_distance",None)
    if camera_distance is not None and (not np.isfinite(camera_distance) or camera_distance<=0):
        raise ValueError("camera-distance must be finite and positive")
    root=Path(args.output).resolve()
    if root.exists() and (not root.is_dir() or any(root.iterdir())):
        raise ValueError("Batch output must be new or empty; choose another --output directory")
    torch.set_num_threads(1)
    model,config,source_provenance=load_policy_snapshot(args.model)
    food_profile=getattr(args,"food_profile",None)
    test_config=apply_food_profile(config,food_profile)
    root.mkdir(parents=True,exist_ok=True)
    snapshot=root/"model.zip"
    save_model(model,snapshot)
    # Load the saved snapshot once so every clip and its fingerprint refer to it.
    loaded=load_policy_snapshot(snapshot)
    manifest={"status":"recording","source_model":source_provenance["path"],"snapshot":"model.zip",
              "source_model_provenance":source_provenance,"snapshot_provenance":loaded[2],
              "training_timesteps":int(model.num_timesteps),"training_config":config.to_dict(),
              "food_counts":args.food_counts,"seeds":args.seeds,
              "food_profile_override":food_profile,"food_geometry":test_config.food_geometry(),
              "camera_distance":camera_distance if camera_distance is not None else 30,
              "planned_videos":len(args.food_counts)*len(args.seeds),"clips":[]}
    write_json(root/"index.json",manifest)
    try:
        for count in args.food_counts:
            for seed in args.seeds:
                name=f"food_{count:02d}_seed_{seed}"
                print(f"VIDEO {len(manifest['clips'])+1}/{manifest['planned_videos']}: {name}",flush=True)
                trial=argparse.Namespace(model=str(snapshot),config=None,seconds=None,seed=seed,
                                         food_count=count,viewer=False,video=str(root/f"{name}.mp4"),
                                         output=str(root/name),camera_distance=camera_distance,
                                         food_profile=food_profile)
                result=rollout_forage(trial,loaded_policy=loaded,show_summary=False)
                manifest["clips"].append({"food_count":count,"seed":seed,"video":f"{name}.mp4",
                                          "summary":f"{name}/summary.json","metrics":result["metrics"],
                                          "food_geometry_differs_from_training":result["food_geometry_differs_from_training"],
                                          "outside_training_food_count_range":result["outside_training_food_count_range"]})
                write_json(root/"index.json",manifest)
                metrics=result["metrics"]
                print(f"SAVED {name}.mp4: consumed={metrics['consumed']:.2f}/{count}, "
                      f"fallen={metrics['fallen']}, duration={metrics['duration_s']:.2f}s",flush=True)
        manifest["status"]="complete"
    except KeyboardInterrupt:
        manifest["status"]="interrupted"
        print("Recording stopped; completed clips and the model snapshot are preserved.",flush=True)
    except Exception as error:
        manifest.update(status="failed",error=str(error))
        raise
    finally:
        write_json(root/"index.json",manifest)
    print(f"Batch {manifest['status']}: {len(manifest['clips'])}/{manifest['planned_videos']} videos in {root}",flush=True)


def dispatch(args):
    if args.command=="forage":
        rollout_forage(args)
    elif args.command=="forage-batch":
        record_batch(args)
    elif args.command=="forage-loop":
        from .forage_learning import run_loop
        run_loop(args)
    elif args.command=="forage-evaluate":
        from .forage_learning import compare_forage
        compare_forage(args)
    elif args.command=="forage-check":
        from .forage_checks import run_checks
        run_checks()
