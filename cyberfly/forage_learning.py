"""Persistent champion/challenger training with explicit, conservative promotion."""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from .behavior import BiologicalReference, describe, no_reference
from .brain import SensoryMemoryEncoder
from .foraging import ForageConfig, ForagingEnv
from .learning import save_model, write_json


def resolve_model(path):
    path = Path(path).resolve()
    if path.is_dir():
        state = json.loads((path/"state.json").read_text())
        path = path/state["champion"]
    return path


def load_policy(path):
    model = PPO.load(resolve_model(path), device="cpu")
    saved = getattr(model,"cyberfly_forage_config",None)
    if saved is None:
        raise ValueError("Expected a foraging checkpoint; walking models use different inputs/actions")
    config = ForageConfig(**saved)
    if config.version != ForageConfig().version:
        raise ValueError("Foraging checkpoint version mismatch")
    return model,config


def evaluate_forage(env, model, seeds, reference=None):
    rows, features = [], []
    for seed in seeds:
        obs,_ = env.reset(seed=seed)
        while True:
            action = np.zeros(3,dtype=np.float32) if model is None else model.predict(obs,deterministic=True)[0]
            obs,_,terminated,truncated,info = env.step(action)
            if terminated or truncated:
                rows.append({"seed":int(seed),**info["metrics"]})
                features.append(describe(env.trajectory))
                break
    return {"episodes":len(rows), "rows":rows, "features":features,
            "mean_consumed_fraction":float(np.mean([r["consumed_fraction"] for r in rows])),
            "ate_any_rate":float(np.mean([r["ate_any"] for r in rows])),
            "ate_all_rate":float(np.mean([r["ate_all"] for r in rows])),
            "fall_rate":float(np.mean([r["fallen"] for r in rows])),
            "escape_rate":float(np.mean([r["escaped"] for r in rows])),
            "biology":reference.compare(features) if reference else no_reference()}


def promotion_decision(incumbent, candidate, min_gain=0.002, bio_min_gain=0.02):
    """Paired bootstrap heuristic, not a guarantee against statistical false positives."""
    old,curr = incumbent["rows"],candidate["rows"]
    if [r["seed"] for r in old] != [r["seed"] for r in curr]:
        raise ValueError("Promotion requires paired scenarios")
    consumption_delta = np.array([b["consumed_fraction"]-a["consumed_fraction"] for a,b in zip(old,curr)])
    delta = np.array([b.get("task_score",b["consumed_fraction"])
                      -a.get("task_score",a["consumed_fraction"]) for a,b in zip(old,curr)])
    reasons = []
    if len(delta)<5:
        reasons.append("fewer_than_five_paired_validation_trials")
    bootstrap = np.random.default_rng(17).choice(delta,(2000,len(delta)),replace=True).mean(axis=1)
    low,high = np.percentile(bootstrap,[2.5,97.5])
    if np.any(consumption_delta < -1e-6):
        reasons.append("consumption_regressed_on_a_paired_trial")
    for key in ("fall_rate","escape_rate"):
        if candidate[key]>incumbent[key]:
            reasons.append(key+"_regressed")
    biological_claim = False
    if incumbent["biology"]["available"] != candidate["biology"]["available"]:
        raise ValueError("Both policies must use the same biological reference")
    if candidate["biology"]["available"]:
        if (incumbent["biology"]["reference_sha256"] != candidate["biology"]["reference_sha256"]):
            raise ValueError("Reference changed during comparison")
        if incumbent["biology"]["distance"]-candidate["biology"]["distance"] <= bio_min_gain:
            reasons.append("no_clear_biological_discrepancy_reduction")
        if low < -min_gain:
            reasons.append("task_score_noninferiority_failed")
        biological_claim = not reasons
    elif float(low) <= min_gain:
        reasons.append("no_clear_task_score_improvement")
    return {"promote":not reasons, "reasons":reasons,
            "mean_consumed_fraction_gain":float(consumption_delta.mean()),
            "mean_task_score_gain":float(delta.mean()),
            "task_score_paired_bootstrap_95_percent_interval":[float(low),float(high)],
            "biological_improvement_supported_on_reference":biological_claim,
            "limitation":"Finite validation data and repeated selection cannot ensure real-world "
                         "or per-round monotonic improvement."}


@contextmanager
def exclusive_run(root):
    root.mkdir(parents=True,exist_ok=True)
    with (root/".train.lock").open("w") as handle:
        try:
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A foraging loop already owns this output directory")
        try:
            yield
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN)


class PeriodicSave(BaseCallback):
    def __init__(self,path):
        super().__init__()
        self.path=path
    def _on_step(self):
        if self.n_calls%256==0:
            save_model(self.model,self.path)
        return True


def new_policy(env,config,seed):
    model=PPO("MlpPolicy",env,device="cpu",seed=seed,verbose=1,
              n_steps=128,batch_size=64,n_epochs=3,learning_rate=1e-4,
              gamma=0.995,ent_coef=0.001,
              policy_kwargs={"features_extractor_class":SensoryMemoryEncoder,
                             "net_arch":[64,64],"log_std_init":-1.5})
    # Deterministic initial policy leaves the working sensory baseline untouched.
    torch.nn.init.zeros_(model.policy.action_net.weight)
    torch.nn.init.zeros_(model.policy.action_net.bias)
    model.cyberfly_forage_config=config.to_dict()
    return model


def run_loop(args):
    if args.steps<128 or args.steps%128 or args.rounds<0 or args.eval_episodes<1:
        raise ValueError("steps must be a positive multiple of 128; rounds>=0; eval-episodes>=1")
    torch.set_num_threads(1)
    root=Path(args.run).resolve()
    with exclusive_run(root):
        state_path=root/"state.json"
        if state_path.exists():
            state=json.loads(state_path.read_text())
            config=ForageConfig(**state["config"])
            if args.config and ForageConfig(**json.loads(Path(args.config).read_text()))!=config:
                raise ValueError("Existing loop configuration cannot change")
        else:
            if any(path.name!=".train.lock" for path in root.iterdir()):
                raise ValueError("New loop requires an empty output directory; existing files preserved")
            config=(ForageConfig(**json.loads(Path(args.config).read_text()))
                    if args.config else ForageConfig())
            state={"schema":1,"config":config.to_dict(),"rounds_completed":0,
                   "champion":"initial.zip","promotions":0}
        reference=BiologicalReference(args.reference,config) if args.reference else None
        digest=reference.digest if reference else None
        if state_path.exists() and state.get("reference_sha256")!=digest:
            raise ValueError("Resume requires the identical biological reference (or none)")
        state["reference_sha256"]=digest
        state["biology_status"]="matched_reference" if reference else "unmeasured"
        env=Monitor(ForagingEnv(config))
        validation_env=ForagingEnv(config)
        try:
            if not state_path.exists():
                initial=new_policy(env,config,args.seed)
                save_model(initial,root/"initial.zip")
                write_json(state_path,state)
                del initial
            count=0
            while args.rounds==0 or count<args.rounds:
                cycle=state["rounds_completed"]+1
                # A previous interrupted attempt is kept, never overwritten.
                attempt=1
                folder=root/f"round_{cycle:04d}_attempt_{attempt:02d}"
                while folder.exists():
                    attempt+=1
                    folder=root/f"round_{cycle:04d}_attempt_{attempt:02d}"
                folder.mkdir()
                champion,_=load_policy(root/state["champion"])
                resume_candidate=None
                if attempt>1:
                    previous=root/f"round_{cycle:04d}_attempt_{attempt-1:02d}"
                    if (previous/"status.json").exists():
                        saved=json.loads((previous/"status.json").read_text())
                        if (saved.get("status")=="interrupted_unvalidated"
                                and saved.get("parent")==state["champion"]
                                and (previous/"candidate.zip").exists()):
                            resume_candidate=previous/"candidate.zip"
                challenger,_=load_policy(resume_candidate or root/state["champion"])
                challenger.set_env(env)
                challenger.set_random_seed(args.seed+cycle)
                challenger.tensorboard_log=str(root/"tb")
                started=time.perf_counter()
                status={"round":cycle,"parent":state["champion"],"status":"training",
                        "config":config.to_dict(),"seed":args.seed+cycle,
                        "resumed_candidate":str(resume_candidate) if resume_candidate else None,
                        "initial_timesteps":int(challenger.num_timesteps)}
                write_json(folder/"status.json",status)
                try:
                    print(f"ROUND {cycle}: train {args.steps} additional steps",flush=True)
                    challenger.learn(args.steps,reset_num_timesteps=False,
                                     callback=PeriodicSave(folder/"candidate.zip"),
                                     tb_log_name=f"round_{cycle:04d}")
                    save_model(challenger,folder/"candidate.zip")
                    seeds=list(range(100_000,100_000+args.eval_episodes))
                    print("Paired validation: incumbent...",flush=True)
                    old=evaluate_forage(validation_env,champion,seeds,reference)
                    print("Paired validation: candidate...",flush=True)
                    new=evaluate_forage(validation_env,challenger,seeds,reference)
                    decision=promotion_decision(old,new)
                    write_json(folder/"evaluation.json",{"incumbent":old,"candidate":new,
                                                         "decision":decision})
                    if decision["promote"]:
                        state["champion"]=str((folder/"candidate.zip").relative_to(root))
                        state["promotions"]+=1
                    state["rounds_completed"]=cycle
                    status.update(status="complete",decision=decision,
                                  wall_seconds=time.perf_counter()-started)
                    write_json(folder/"status.json",status)
                    # The atomic pointer update is the only commit of a new champion.
                    write_json(state_path,state)
                    print(json.dumps(decision,ensure_ascii=False),flush=True)
                    count+=1
                except KeyboardInterrupt:
                    save_model(challenger,folder/"candidate.zip")
                    status["status"]="interrupted_unvalidated"
                    write_json(folder/"status.json",status)
                    print("Saved unvalidated candidate; champion unchanged. Restart the same command to continue.",flush=True)
                    break
                except Exception as error:
                    status.update(status="failed",error=str(error))
                    write_json(folder/"status.json",status)
                    raise
        finally:
            env.close()
            validation_env.close()


def compare_forage(args):
    torch.set_num_threads(1)
    model,config=load_policy(args.model)
    reference=BiologicalReference(args.reference,config) if args.reference else None
    if args.episodes<1:
        raise ValueError("episodes must be positive")
    env=ForagingEnv(config)
    try:
        seeds=list(range(args.seed_start,args.seed_start+args.episodes))
        baseline=evaluate_forage(env,None,seeds,reference)
        learned=evaluate_forage(env,model,seeds,reference)
        write_json(args.output,{"baseline":baseline,"candidate":learned,
                                 "model_timesteps":int(model.num_timesteps),
                                 "model":str(resolve_model(args.model))})
        for name,stats in (("baseline",baseline),("candidate",learned)):
            print(name,{k:stats[k] for k in ("mean_consumed_fraction","ate_any_rate",
                                            "fall_rate","escape_rate","biology")},flush=True)
    finally:
        env.close()
