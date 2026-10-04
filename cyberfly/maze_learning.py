"""Cumulative PPO maze training with measured CPU/Apple MPS selection."""
import hashlib
from io import BytesIO
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from .forage_learning import exclusive_run, resolve_model
from .learning import save_model, write_json
from .maze import MazeConfig
from .maze_brain import MazeMemoryEncoder
from .maze_env import MazeEnv
from .training_device import require_device, policy_device_report, select_training_device


def load_maze_policy(path, device="cpu"):
    checkpoint = resolve_model(path)
    payload = checkpoint.read_bytes()
    with BytesIO(payload) as stream:
        model = PPO.load(stream,device=device)
    saved = getattr(model,"cyberfly_maze_config",None)
    if saved is None:
        raise ValueError("Expected a maze checkpoint; walking and open-field policies are incompatible")
    config = MazeConfig(**saved)
    return model,config,{"path":str(checkpoint),"sha256":hashlib.sha256(payload).hexdigest(),"size_bytes":len(payload)}


def new_maze_policy(env, config, seed, device, rollout_steps):
    model = PPO("MlpPolicy",env,device=device,seed=seed,verbose=1,n_steps=rollout_steps,
                batch_size=min(256,rollout_steps*env.num_envs),n_epochs=4,learning_rate=1e-4,
                gamma=config.gamma,gae_lambda=0.95,ent_coef=0.005,
                policy_kwargs={"features_extractor_class":MazeMemoryEncoder,
                               "net_arch":[256,128],"activation_fn":torch.nn.ReLU,"log_std_init":-1.8})
    torch.nn.init.zeros_(model.policy.action_net.weight)
    torch.nn.init.zeros_(model.policy.action_net.bias)
    model.cyberfly_maze_config = config.to_dict()
    policy_device_report(model)
    return model


def make_worker(config):
    def build():
        torch.set_num_threads(1)
        return Monitor(MazeEnv(config))
    return build


def evaluate_maze(env, model, seeds):
    rows = []
    for seed in seeds:
        obs,reset_info = env.reset(seed=seed)
        while True:
            action = np.zeros(4,dtype=np.float32) if model is None else model.predict(obs,deterministic=True)[0]
            obs,_,terminated,truncated,info = env.step(action)
            if terminated or truncated:
                rows.append({"seed":seed,"maze":reset_info["maze"],**info["metrics"]})
                break
    return {"episodes":len(rows),"rows":rows,
            "success_rate":float(np.mean([row["ate_all"] for row in rows])),
            "mean_consumed_fraction":float(np.mean([row["consumed_fraction"] for row in rows])),
            "mean_return":float(np.mean([row["return"] for row in rows])),
            "fall_rate":float(np.mean([row["fallen"] for row in rows])),
            "mean_duration_s":float(np.mean([row["duration_s"] for row in rows])),
            "biological_similarity":None}


def selection_score(stats):
    # Finite validation selection, not a claim of statistically established improvement.
    return (stats["success_rate"],stats["mean_consumed_fraction"],-stats["fall_rate"],
            -stats["mean_duration_s"],stats["mean_return"])


def train_maze(args):
    if not (1 <= args.envs <= 8 and args.rollout_steps >= 64 and args.rollout_steps%64 == 0):
        raise ValueError("Use 1–8 environments and rollout-steps a positive multiple of 64")
    block = args.envs*args.rollout_steps
    if args.seed < 0 or args.steps < 0 or args.steps%block or args.eval_every < block or args.eval_every%block or args.eval_episodes < 1:
        raise ValueError("steps (0 = continuous) and eval-every must be multiples of envs * rollout-steps; eval-episodes >= 1")
    require_device("cpu" if args.device == "auto" else args.device)
    root = Path(args.run).resolve()
    with exclusive_run(root):
        state_path = root/"state.json"
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state.get("task") != "maze":
                raise ValueError("Use a separate run directory for maze training")
            config = MazeConfig(**state["config"])
            if args.config and MazeConfig(**json.loads(Path(args.config).read_text())) != config:
                raise ValueError("Cannot change the configuration of an existing maze run")
            if (state["envs"],state["rollout_steps"],state["seed"]) != (args.envs,args.rollout_steps,args.seed):
                raise ValueError("Resume with the same envs, rollout-steps and seed")
            if state["best_evaluation"] is not None and state["best_evaluation"]["episodes"] != args.eval_episodes:
                raise ValueError("Resume with the same eval-episodes to preserve comparable validation")
        else:
            if any(p.name != ".train.lock" for p in root.iterdir()):
                raise ValueError("A new maze run requires an empty directory")
            config = MazeConfig(**json.loads(Path(args.config).read_text())) if args.config else MazeConfig()
            state = {"schema":1,"task":"maze","config":config.to_dict(),"envs":args.envs,
                     "rollout_steps":args.rollout_steps,"seed":args.seed,"latest":"latest.zip",
                     "champion":"initial.zip","best_evaluation":None,"evaluations":0}
        env = validation = model = None
        try:
            builders = [make_worker(config) for _ in range(args.envs)]
            env = DummyVecEnv(builders) if args.envs == 1 else SubprocVecEnv(builders,start_method="spawn")
            validation = MazeEnv(config)
            if state_path.exists():
                model,_,_ = load_maze_policy(root/state["latest"],"cpu")
                model.set_env(env)
            else:
                model = new_maze_policy(env,config,args.seed,"cpu",args.rollout_steps)
                save_model(model,root/"initial.zip")
                save_model(model,root/"latest.zip")
            device_info = select_training_device(model,env,args.device)
            if device_info["policy_device"] != model.device.type:
                snapshot = BytesIO()
                model.save(snapshot)
                snapshot.seek(0)
                model = PPO.load(snapshot,env=env,device=device_info["policy_device"])
            policy_device_report(model)
            print(json.dumps({"device_check":device_info},ensure_ascii=False),flush=True)
            model.set_random_seed(args.seed+model.num_timesteps)
            model.tensorboard_log = str(root/"tb")
            state.pop("error",None)
            state.update(device=device_info,status="training",training_timesteps=int(model.num_timesteps))
            write_json(state_path,state)
            start = int(model.num_timesteps)
            target = start+args.steps if args.steps else None
            while target is None or model.num_timesteps < target:
                steps = args.eval_every if target is None else min(args.eval_every,target-model.num_timesteps)
                started = time.perf_counter()
                model.learn(total_timesteps=steps,reset_num_timesteps=False,tb_log_name="maze")
                device_report = policy_device_report(model)
                if device_report["gradient_tensors"] == 0:
                    raise RuntimeError("PPO did not produce trainable gradients")
                checkpoint = root/"checkpoints"/f"step_{model.num_timesteps:010d}.zip"
                save_model(model,checkpoint)
                save_model(model,root/"latest.zip")
                state.update(training_timesteps=int(model.num_timesteps),last_device_report=device_report,
                             ppo_epochs_completed=int(model._n_updates),
                             training_wall_seconds=time.perf_counter()-started,status="evaluating")
                write_json(state_path,state)
                print("Validating fixed held-out maze seeds...",flush=True)
                stats = evaluate_maze(validation,model,list(range(200000,200000+args.eval_episodes)))
                stats.update(training_timesteps=int(model.num_timesteps),device_report=device_report,
                             model_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
                write_json(checkpoint.with_suffix(".evaluation.json"),stats)
                old = state["best_evaluation"]
                if old is None or selection_score(stats) > selection_score(old):
                    state.update(champion=str(checkpoint.relative_to(root)),best_evaluation=stats)
                state.update(status="training",evaluations=state["evaluations"]+1)
                write_json(state_path,state)
                print(json.dumps({"timesteps":model.num_timesteps,"success_rate":stats["success_rate"],
                                  "mean_return":stats["mean_return"],"device":device_report},ensure_ascii=False),flush=True)
            state["status"] = "complete"
        except KeyboardInterrupt:
            state["status"] = "interrupted_unvalidated_latest"
            print("Stopped. Latest learning state is saved; rerun the same command to continue.",flush=True)
        except Exception as error:
            state.update(status="failed",error=str(error))
            raise
        finally:
            if model is not None:
                save_model(model,root/"latest.zip")
                state["training_timesteps"] = int(model.num_timesteps)
                write_json(state_path,state)
            if validation is not None:
                validation.close()
            if env is not None:
                env.close()
