"""Sensory-driven terrestrial sugar foraging, with explicit modeling limits."""
from collections import deque
from dataclasses import asdict, dataclass, replace

import gymnasium as gym
import mujoco
import numpy as np
from flygym.anatomy import BodySegment, LEGS
from flygym.utils.mjcf import add_actuator
from flygym_demo.complex_terrain import HybridControllerObservation, LocomotionAction, apply_locomotion_action

from .brain import EYE_ANGLES, FRAME_SIZE, SensoryBrain
from .config_validation import require_time_multiple, validate_numbers
from .env import CyberflyEnv, TaskConfig

FOOD_PROFILES = ("legacy-patch", "sucrose-droplet")
DROPLET_SOURCE = "https://www.frontiersin.org/journals/behavioral-neuroscience/articles/10.3389/fnbeh.2018.00280/full"


@dataclass(frozen=True)
class ForageConfig:
    episode_seconds: float = 15.0
    food_min: int = 1
    food_max: int = 5
    food_radius: float = 0.8
    spawn_min: float = 4.0
    spawn_max: float = 12.0
    arena_radius: float = 25.0
    visual_range: float = 22.0
    sensor_noise: float = 0.002
    sip_seconds: float = 0.16
    ingestion_rate: float = 2.0
    volatile_food_odor: bool = False
    residual_scale: float = 0.15
    # None keeps the historical shared scale, including custom old checkpoints.
    mouth_residual_scale: float | None = None
    # Legacy defaults preserve the geometry of existing checkpoints.
    food_profile: str = "legacy-patch"
    droplet_volume_ul: float = 0.1
    version: int = 1

    def __post_init__(self):
        validate_numbers(self, integers=("food_min", "food_max", "version"),
                         optional=("mouth_residual_scale",),
                         exclude=("food_profile", "volatile_food_odor"))
        if type(self.volatile_food_odor) is not bool:
            raise ValueError("volatile_food_odor must be a boolean")
        if self.version != 1:
            raise ValueError("Unsupported foraging configuration version")
        if self.food_profile not in FOOD_PROFILES:
            raise ValueError("Unknown food profile")
        if not 1 <= self.food_min <= self.food_max <= 12:
            raise ValueError("Food count must be between 1 and 12")
        if not 0 < self.spawn_min < self.spawn_max < self.arena_radius:
            raise ValueError("Invalid food/arena radii")
        if min(self.food_radius, self.sip_seconds, self.ingestion_rate,
               self.episode_seconds, self.visual_range, self.droplet_volume_ul) <= 0:
            raise ValueError("Durations, radii and rates must be positive")
        if self.sensor_noise < 0 or not 0 <= self.residual_scale <= 0.5:
            raise ValueError("Invalid noise/residual scale")
        if self.mouth_residual_scale is not None and not 0 <= self.mouth_residual_scale <= 2:
            raise ValueError("mouth_residual_scale must be between 0 and 2")
        require_time_multiple(self.episode_seconds, TaskConfig().control_dt)
        if self.food_profile == "sucrose-droplet" and self.food_half_height > self.food_radius:
            raise ValueError("Droplet volume/radius must describe a flattened ellipsoid")

    @property
    def food_half_height(self):
        # FlyGym length is mm; 1 mm^3 = 1 microlitre. V = 4*pi*a*b*c/3.
        if self.food_profile == "sucrose-droplet":
            return 3*self.droplet_volume_ul/(4*np.pi*self.food_radius**2)
        return 0.12

    def food_geometry(self):
        droplet = self.food_profile == "sucrose-droplet"
        return {"profile":self.food_profile,"shape":"ellipsoid" if droplet else "cylinder",
                "diameter_mm":2*self.food_radius,"height_mm":2*self.food_half_height,
                "geometric_volume_ul":self.droplet_volume_ul if droplet else
                                      np.pi*self.food_radius**2*2*self.food_half_height,
                "volume_source":DROPLET_SOURCE if droplet else None,
                "shape_status":("Engineering approximation; droplet footprint/contact angle not measured"
                                if droplet else "Legacy abstract nutrient patch"),
                "intake_units":"Normalized food units; ingestion rate and satiety are not volume-calibrated"}

    def to_dict(self):
        data = asdict(self)
        # Preserve exact legacy configuration manifests and saved run metadata.
        if self.mouth_residual_scale is None:
            del data["mouth_residual_scale"]
        return data

    def motor_command(self, base, action):
        motor = base + self.residual_scale*action
        if self.mouth_residual_scale is not None:
            motor[2] = base[2] + self.mouth_residual_scale*action[2]
        return np.clip(motor, -1, 1)


def apply_food_profile(config, profile):
    """Select a documented geometry preset without changing a stored policy."""
    if profile is None:
        return config
    if profile not in FOOD_PROFILES:
        raise ValueError("Unknown food profile")
    return replace(config,food_profile=profile,
                   food_radius=0.4 if profile=="sucrose-droplet" else 0.8,
                   droplet_volume_ul=0.1)


class ForagingEnv(CyberflyEnv):
    """PPO actions are bounded corrections to the sensory-only reflex baseline.

Food locations are simulator state, used for rendering, sensor generation and
contact/ingestion accounting, never passed as coordinates to the actor/brain.
"""
    def __init__(self, config=None, render_mode=None):
        self.forage_config = config or ForageConfig()
        super().__init__(TaskConfig(episode_seconds=self.forage_config.episode_seconds,
                                    heading_range=0.0), render_mode)
        # Share a stationary arena camera between the viewer and video renderer.
        # Following the thorax would turn walking/body oscillations into camera shake.
        self.camera_follow_body = False
        self.camera.distance = 30
        self.camera.azimuth = 100
        self.camera.elevation = -65
        self.camera.lookat[:] = [0, 0, 0]
        self.action_space = gym.spaces.Box(-1, 1, (3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(-10, 10, (4, FRAME_SIZE), dtype=np.float32)
        self.brain = SensoryBrain()
        self.history = deque(maxlen=4)
        self.mouth_actuator = self.sim.mj_model.actuator("cyberfly/proboscis_servo").id
        self.mouth_joint = self.sim.mj_model.joint("cyberfly/proboscis_extension").id
        self.mouth_qpos = self.sim.mj_model.jnt_qposadr[self.mouth_joint]
        self.mouth_site = self.sim.mj_model.site("cyberfly/feeding_tip").id
        self.food_mocap = [self.sim.mj_model.body(f"sugar_{i}").mocapid[0]
                           for i in range(self.forage_config.food_max)]
        self.food_geoms = [self.sim.mj_model.geom(f"sugar_geom_{i}").id
                          for i in range(self.forage_config.food_max)]
        order = self.fly.get_bodysegs_order()
        self.foot_indices = [order.index(BodySegment(f"{leg}_tarsus5")) for leg in LEGS]
        self.eye_sites = [self.sim.mj_model.site(f"cyberfly/sense_eye_{s}").id
                          for s in ("l", "r")]
        self.antenna_sites = [self.sim.mj_model.site(f"cyberfly/sense_antenna_{s}").id
                              for s in ("l", "r")]

    def _configure_fly(self):
        # One actuated slide moves the existing rostrum/haustellum mesh. This is
        # functional extension, not a validated multi-muscle mouth model.
        rostrum = self.fly.bodyseg_to_mjcfbody[BodySegment("c_rostrum")]
        rostrum.add_joint(name="proboscis_extension", type=mujoco.mjtJoint.mjJNT_SLIDE,
                          axis=[0, 0, -1], limited=True, range=[0, 0.65],
                          damping=0.02, armature=0.0001)
        add_actuator(self.fly.mjcf_root, "position", name="proboscis_servo",
                     joint="proboscis_extension", kp=2.0, kv=0.05,
                     ctrllimited=True, ctrlrange=[0, 0.65],
                     forcelimited=True, forcerange=[-5, 5])
        mouth = self.fly.bodyseg_to_mjcfbody[BodySegment("c_haustellum")]
        mouth.add_site(name="feeding_tip", pos=[0.45, 0, -0.2], size=[0.05]*3,
                       rgba=[0.2, 0.9, 0.5, 1])
        for side, sign in (("l", 1), ("r", -1)):
            eye = self.fly.bodyseg_to_mjcfbody[BodySegment(f"{side}_eye")]
            eye.add_site(name=f"sense_eye_{side}", pos=[0, sign*0.15, 0],
                         size=[0.015]*3, rgba=[0, 0, 0, 0])
            antenna = self.fly.bodyseg_to_mjcfbody[BodySegment(f"{side}_funiculus")]
            antenna.add_site(name=f"sense_antenna_{side}", size=[0.015]*3,
                             rgba=[0, 0, 0, 0])

    def _configure_world(self, world):
        cfg = self.forage_config
        droplet = cfg.food_profile == "sucrose-droplet"
        for i in range(self.forage_config.food_max):
            body = world.mjcf_root.worldbody.add_body(name=f"sugar_{i}",
                                                      mocap=True, pos=[0, 0, -100])
            shape = mujoco.mjtGeom.mjGEOM_ELLIPSOID if droplet else mujoco.mjtGeom.mjGEOM_CYLINDER
            size = ([cfg.food_radius,cfg.food_radius,cfg.food_half_height] if droplet else
                    [cfg.food_radius,cfg.food_half_height,0])
            body.add_geom(name=f"sugar_geom_{i}", type=shape, size=size,
                          rgba=[0.95, 0.88, 0.66, 1], contype=0, conaffinity=0,
                          group=0)

    @property
    def mouth_position(self):
        return self.sim.mj_data.site_xpos[self.mouth_site].copy()

    def _near(self, point, horizontal_extra=0.0, vertical=0.25):
        if not hasattr(self, "food_remaining"):
            return np.zeros(self.forage_config.food_max, dtype=bool)
        if self.forage_config.food_profile == "sucrose-droplet":
            # Approximate contact with the visible ellipsoid. A small mouth/foot
            # tolerance replaces the legacy patch's broad vertical contact slab.
            cfg = self.forage_config
            axes = np.array([cfg.food_radius+horizontal_extra,cfg.food_radius+horizontal_extra,
                             cfg.food_half_height+min(vertical,0.05)])
            return ((np.sum(((self.food_positions-point)/axes)**2,axis=1)<=1)
                    & (self.food_remaining>1e-8))
        return ((np.linalg.norm(self.food_positions[:, :2] - point[:2], axis=1)
                 <= self.forage_config.food_radius + horizontal_extra)
                & (abs(point[2] - 0.12) <= vertical)
                & (self.food_remaining > 1e-8))

    def _sensors(self):
        cfg = self.forage_config
        visual = np.zeros((2, 24))
        active = self.food_remaining > 1e-8
        for eye_i, site in enumerate(self.eye_sites):
            origin = self.sim.mj_data.site_xpos[site]
            delta = self.food_positions - origin
            distances = np.linalg.norm(delta[:, :2], axis=1)
            bearings = np.arctan2(delta[:, 1], delta[:, 0]) - self.heading
            for distance, bearing, alive in zip(distances, bearings, active):
                if not alive or distance > cfg.visual_range:
                    continue
                radius = np.arctan2(cfg.food_radius, max(distance, 0.05))
                angular = np.arctan2(np.sin(EYE_ANGLES[eye_i]-bearing),
                                     np.cos(EYE_ANGLES[eye_i]-bearing))
                # Low resolution contrast/size retina of known food patches.
                # No RGB optics, photoreceptor dynamics or object classification.
                response = min(1.0, radius) * np.exp(-0.5*(angular/max(0.07, radius))**2)
                visual[eye_i] = np.maximum(visual[eye_i], response)
        visual += self.np_random.normal(0, cfg.sensor_noise, visual.shape)
        visual = np.clip(visual, 0, 1)
        odor = np.zeros(2)
        # Pure sugar does not provide this optional volatile-food cue.
        if cfg.volatile_food_odor:
            for side, site in enumerate(self.antenna_sites):
                distance = np.linalg.norm(self.food_positions - self.sim.mj_data.site_xpos[site], axis=1)
                odor[side] = np.sum(np.exp(-distance/8.0)*active) / cfg.food_max
        feet = self.sim.get_body_positions(self.fly.name)[self.foot_indices]
        taste = np.array([self._near(point, 0.12, 0.3).any() for point in feet], dtype=float)
        mouth_taste = float(self._near(self.mouth_position, 0.05, 0.25).any())
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.sim.mj_model, self.sim.mj_data,
                                mujoco.mjtObj.mjOBJ_BODY, self.thorax_id, velocity, 1)
        cpg = self.controller.cpg_network
        frame = np.concatenate((
            visual.ravel(), odor, taste, [mouth_taste, self.energy],
            (self.rotation.T @ self.filtered_velocity)/20,
            velocity[:3]/20, self.rotation[2, :], np.sin(cpg.curr_phases),
            np.cos(cpg.curr_phases), cpg.curr_magnitudes,
            [self.sim.mj_data.qpos[self.mouth_qpos]/0.65], self.last_action,
            [self.recent_food],
        )).astype(np.float32)
        if frame.shape != (FRAME_SIZE,) or not np.isfinite(frame).all():
            raise FloatingPointError("Invalid sensory frame")
        return np.clip(frame, -10, 10)

    def _observation(self):
        # Base reset calls this before the foraging state has been initialized.
        return np.zeros((4, FRAME_SIZE), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        cfg = self.forage_config
        self.brain.reset()
        self.energy = 0.45
        self.recent_food = 0.0
        self.last_action = np.zeros(3, dtype=np.float32)
        self.last_command = LocomotionAction(
            self.steps.default_pose_by_dof_order(self.dof_order), np.ones(6, dtype=bool))
        self.sip_elapsed = 0.0
        self.sip_target = None
        self.consumed = 0.0
        self.feeding_time = 0.0
        self.first_ingestion_time = None
        self.path_length = 0.0
        self.trajectory = []
        self.events = []
        self.food_positions = np.zeros((cfg.food_max, 3))
        self.food_positions[:, 2] = cfg.food_half_height
        self.food_remaining = np.zeros(cfg.food_max)
        provided = (options or {}).get("food_positions")
        if provided is not None:
            provided = np.asarray(provided, dtype=float)
            if (provided.ndim != 2 or provided.shape[1] != 2
                    or not 1 <= len(provided) <= cfg.food_max or not np.isfinite(provided).all()):
                raise ValueError("food_positions must be a finite N x 2 array")
            self.food_count = len(provided)
            self.food_positions[:self.food_count, :2] = provided
        else:
            self.food_count = int(self.np_random.integers(cfg.food_min, cfg.food_max+1))
            for i in range(self.food_count):
                for _ in range(1000):
                    radius = np.sqrt(self.np_random.uniform(cfg.spawn_min**2, cfg.spawn_max**2))
                    angle = self.np_random.uniform(-np.pi, np.pi)
                    point = self.position[:2] + radius*np.array([np.cos(angle), np.sin(angle)])
                    if i == 0 or np.all(np.linalg.norm(self.food_positions[:i,:2]-point,axis=1)
                                        > 2*cfg.food_radius+0.5):
                        self.food_positions[i,:2] = point
                        break
                else:
                    raise RuntimeError("Cannot place separated food patches")
        self.food_remaining[:self.food_count] = 1.0
        self._sync_food()
        mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)
        self.history.clear()
        self.frame = self._sensors()
        self.history.extend([self.frame.copy() for _ in range(4)])
        self.trajectory.append({"time_s":0.0,"x_mm":float(self.position[0]),
                                "y_mm":float(self.position[1]),"heading_rad":self.heading,
                                "speed_mm_s":0.0,"filtered_speed_mm_s":0.0,"feeding":0,
                                "mouth_extension_mm":float(self.sim.mj_data.qpos[self.mouth_qpos]),
                                "energy":self.energy,"consumed":0.0,"mode":"search"})
        self.episode_return = 0.0
        return np.stack(self.history), {"food_count": self.food_count}

    def _sync_food(self):
        for i, mocap in enumerate(self.food_mocap):
            active = self.food_remaining[i] > 1e-8
            self.sim.mj_data.mocap_pos[mocap] = self.food_positions[i] if active else [0, 0, -100]
            self.sim.mj_model.geom_rgba[self.food_geoms[i], 3] = float(active)

    def _ingest(self, mouth_command, speed):
        touching = np.flatnonzero(self._near(self.mouth_position, 0.05, 0.25))
        ready = (mouth_command > 0.5 and len(touching) > 0 and speed < 3.0
                 and self.energy < 0.98 and self.rotation[2,2] > 0.7)
        if not ready:
            self.sip_elapsed = 0.0
            self.sip_target = None
            return 0.0, False
        target = int(touching[0])
        if self.sip_target != target:
            self.sip_elapsed = 0.0
            self.sip_target = target
        self.sip_elapsed += self.config.control_dt
        self.feeding_time += self.config.control_dt
        if self.sip_elapsed + 1e-9 < self.forage_config.sip_seconds:
            return 0.0, True
        self.sip_elapsed = 0.0
        amount = min(self.food_remaining[target], self.forage_config.ingestion_rate
                     * self.forage_config.sip_seconds)
        self.food_remaining[target] = max(0.0, self.food_remaining[target]-amount)
        self.consumed += amount
        self.energy = min(1.0, self.energy + 0.12*amount)
        self.recent_food = 1.0
        if self.first_ingestion_time is None:
            self.first_ingestion_time = self.elapsed_steps*self.config.control_dt
        self.events.append({"time_s": self.elapsed_steps*self.config.control_dt,
                            "food_id": target, "amount": amount,
                            "remaining": float(self.food_remaining[target]),
                            "mouth_xyz_mm": self.mouth_position.tolist()})
        self._sync_food()
        return amount, True

    def _is_escaped(self):
        return bool(np.linalg.norm(self.position[:2]-self.start_position[:2])
                    > self.forage_config.arena_radius)

    def step(self, action):
        if self.done:
            raise RuntimeError("Reset after a finished episode")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (3,) or not np.isfinite(action).all():
            raise ValueError("Expected three finite residual actions")
        action = np.clip(action, -1, 1)
        base = self.brain.act(self.frame, self.config.control_dt)
        motor = self.forage_config.motor_command(base, action)
        previous = self.position
        before = self.sim.mj_data.time
        self.sim.mj_data.ctrl[self.mouth_actuator] = 0.4*max(0.0, motor[2])
        for _ in range(self.substeps):
            if np.max(np.abs(motor[:2])) < 0.05:
                # Halt walking while holding the current leg posture and adhesion.
                command = LocomotionAction(self.last_command.joint_angles,
                                           np.ones(6, dtype=bool))
            else:
                sensory = HybridControllerObservation.from_sim(self.sim, self.fly.name)
                command = self.controller.step(motor[:2], sensory)
                self.last_command = command
            apply_locomotion_action(self.sim, self.fly.name, command)
            self.sim.step()
        mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)
        if not np.isfinite(self.sim.mj_data.qpos).all() or self.sim.mj_data.time < before:
            raise FloatingPointError("Invalid foraging simulation state")
        self.elapsed_steps += 1
        dt = self.config.control_dt
        velocity = (self.position - previous)/dt
        self.filtered_velocity += (1-np.exp(-dt/0.1))*(velocity-self.filtered_velocity)
        speed = float(np.linalg.norm(self.filtered_velocity[:2]))
        self.path_length += float(np.linalg.norm(self.position[:2]-previous[:2]))
        self.energy = max(0.0, self.energy - dt*(0.002 + 0.0002*speed))
        self.recent_food *= np.exp(-dt/2)
        amount, feeding = self._ingest(motor[2], speed)
        fallen = bool(self.rotation[2,2] < 0.3 or self.position[2] < 0.3)
        escaped = self._is_escaped()
        finished = bool(np.all(self.food_remaining[:self.food_count] <= 1e-8))
        reward = 5.0*amount - dt*0.02 - dt*0.01*float(np.mean(action**2))
        if fallen or escaped:
            reward -= 2.0
        self.last_action = motor.astype(np.float32)
        self.frame = self._sensors()
        self.history.append(self.frame.copy())
        self.episode_return += reward
        self.done = fallen or escaped or finished or self.energy <= 0 or self.elapsed_steps >= self.max_steps
        row = {"time_s": self.elapsed_steps*dt, "x_mm": float(self.position[0]),
               "y_mm": float(self.position[1]), "heading_rad": self.heading,
               "speed_mm_s": float(np.linalg.norm(velocity[:2])),
               "filtered_speed_mm_s": speed, "feeding": int(feeding),
               "mouth_extension_mm": float(self.sim.mj_data.qpos[self.mouth_qpos]),
               "energy": self.energy, "consumed": self.consumed,
               "mode": self.brain.mode}
        self.trajectory.append(row)
        info = {**row, "fallen": fallen, "escaped": escaped,
                "food_count": self.food_count, "food_remaining": int(np.sum(self.food_remaining>1e-8))}
        if self.done:
            info["metrics"] = {
                "schema_version": 2,
                "return": float(self.episode_return), "consumed": float(self.consumed),
                "food_count": self.food_count,
                "consumed_fraction": float(self.consumed/self.food_count),
                "ate_any": bool(self.consumed>0), "ate_all": finished,
                "fallen": fallen, "escaped": escaped, "duration_s": row["time_s"],
                "first_ingestion_s": self.first_ingestion_time,
                "feeding_fraction": self.feeding_time/max(dt,row["time_s"]),
                "path_length_mm": self.path_length,
                "mean_speed_mm_s": self.path_length/row["time_s"],
                "mean_filtered_speed_mm_s": float(np.mean([
                    r["filtered_speed_mm_s"] for r in self.trajectory[1:]])),
                "task_score": float(self.consumed/self.food_count
                                    -0.02*row["time_s"]/self.config.episode_seconds
                                    -0.0005*self.path_length),
                "biological_similarity": None,
            }
        return (np.stack(self.history), float(reward),
                bool(fallen or escaped or finished or self.energy<=0),
                bool(self.elapsed_steps>=self.max_steps), info)
