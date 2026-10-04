"""Gymnasium task: track a planar velocity command with a hybrid fly controller.

FlyGym model units are mm, mg, seconds. Policy actions modulate two CPG
amplitudes; they do not replace the 42-DoF low-level controller.
"""
from dataclasses import asdict, dataclass

import gymnasium as gym
import mujoco
import numpy as np
from flygym import Simulation
from flygym.anatomy import BodySegment, ContactBodiesPreset
from flygym.compose import FlatGroundWorld
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import (
    HybridControllerObservation, HybridTurningController, LocomotionAction,
    PreprogrammedSteps, apply_locomotion_action, make_locomotion_fly,
)

from .config_validation import require_time_multiple, validate_numbers


@dataclass(frozen=True)
class TaskConfig:
    physics_dt: float = 0.0001
    control_dt: float = 0.02
    episode_seconds: float = 2.0
    speed_min: float = 8.0
    speed_max: float = 12.0
    heading_range: float = 0.35
    action_scale: float = 0.4
    version: int = 1

    def __post_init__(self):
        validate_numbers(self)
        if self.version != 1:
            raise ValueError("Unsupported walking configuration version")
        if min(self.physics_dt, self.control_dt, self.episode_seconds) <= 0:
            raise ValueError("Timesteps and episode duration must be positive")
        for value, unit in ((self.control_dt, self.physics_dt),
                            (self.episode_seconds, self.control_dt)):
            require_time_multiple(value, unit)
        if not 0 < self.speed_min <= self.speed_max:
            raise ValueError("Invalid target speed range")
        if self.heading_range < 0 or not 0 <= self.action_scale <= 1:
            raise ValueError("heading_range must be nonnegative and action_scale between 0 and 1")

    def to_dict(self):
        return asdict(self)


class CyberflyEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 25}

    def __init__(self, config=None, render_mode=None):
        super().__init__()
        self.config = config or TaskConfig()
        if render_mode not in (None, "rgb_array"):
            raise ValueError("Use CLI --viewer for a native window")
        self.render_mode = render_mode
        self.fly = make_locomotion_fly(name="cyberfly", add_adhesion=True,
                                       colorize=True)
        self._configure_fly()
        world = FlatGroundWorld()
        self._configure_world(world)
        world.add_fly(
            self.fly, [0, 0, 1.2], Rotation3D("quat", [1, 0, 0, 0]),
            bodysegs_with_ground_contact=ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD,
            add_ground_contact_sensors=False,
        )
        self.sim = Simulation(world, timestep=self.config.physics_dt)
        self.steps = PreprogrammedSteps()
        self.dof_order = self.fly.get_actuated_jointdofs_order("position")
        self.controller = HybridTurningController(
            timestep=self.sim.timestep, preprogrammed_steps=self.steps,
            output_dof_order=self.dof_order,
        )
        thorax = self.fly.bodyseg_to_mjcfbody[BodySegment("c_thorax")].name
        self.thorax_id = self.sim.mj_model.body(thorax).id
        self.substeps = round(self.config.control_dt / self.sim.timestep)
        self.max_steps = round(self.config.episode_seconds / self.config.control_dt)
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        # Body velocity(3), angular velocity(3), world-up in body frame(3),
        # heading error sin/cos(2), target speed(1), height(1), CPG sin/cos(12),
        # CPG magnitudes(6), previous action(2). Scales are fixed, not learned.
        self.observation_space = gym.spaces.Box(-np.inf, np.inf,
                                               shape=(33,), dtype=np.float32)
        self.renderer = None
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 12
        self.camera.azimuth = 135
        self.camera.elevation = -25
        self.camera_follow_body = True
        self.elapsed_steps = 0
        self.done = True

    def _configure_fly(self):
        """Extension hook before model compilation; walking task has no additions."""

    def _configure_world(self, world):
        """Extension hook before attaching the fly and rebuilding keyframes."""

    @property
    def position(self):
        return self.sim.mj_data.xpos[self.thorax_id].copy()

    @property
    def rotation(self):
        return self.sim.mj_data.xmat[self.thorax_id].reshape(3, 3).copy()

    @property
    def heading(self):
        rotation = self.rotation
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def _observation(self):
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(self.sim.mj_model, self.sim.mj_data,
                                mujoco.mjtObj.mjOBJ_BODY, self.thorax_id,
                                velocity, 1)
        error = self.target_heading - self.heading
        cpg = self.controller.cpg_network
        return np.concatenate((
            (self.rotation.T @ self.filtered_velocity) / 20,
            velocity[:3] / 20, self.rotation[2, :],
            [np.sin(error), np.cos(error), self.target_speed / 20,
             self.position[2] / 2],
            np.sin(cpg.curr_phases), np.cos(cpg.curr_phases),
            cpg.curr_magnitudes, self.last_action,
        )).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.sim.reset()
        self.controller.reset(seed=int(self.np_random.integers(0, 2**31 - 1)))
        initial = LocomotionAction(
            self.steps.default_pose_by_dof_order(self.dof_order),
            np.ones(6, dtype=bool),
        )
        apply_locomotion_action(self.sim, self.fly.name, initial)
        self.sim.warmup()
        mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)
        options = options or {}
        self.target_speed = float(options.get("speed", self.np_random.uniform(
            self.config.speed_min, self.config.speed_max)))
        offset = float(options.get("heading_offset", self.np_random.uniform(
            -self.config.heading_range, self.config.heading_range)))
        self.target_heading = self.heading + offset
        self.target_velocity = self.target_speed * np.array([
            np.cos(self.target_heading), np.sin(self.target_heading)])
        self.last_action = np.zeros(2, dtype=np.float32)
        self.filtered_velocity = np.zeros(3)
        self.elapsed_steps = 0
        self.done = False
        self.start_position = self.position
        self.tracking_errors = []
        self.heading_errors = []
        self.action_changes = []
        self.episode_return = 0.0
        return self._observation(), {
            "target_speed_mm_s": self.target_speed,
            "target_heading_rad": self.target_heading,
        }

    def step(self, action):
        if self.done:
            raise RuntimeError("Call reset before stepping a finished episode")
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (2,) or not np.isfinite(action).all():
            raise ValueError("Action must contain two finite values")
        action = np.clip(action, -1, 1)
        drive = 1.0 + self.config.action_scale * action
        previous_position = self.position
        initial_time = self.sim.mj_data.time
        for _ in range(self.substeps):
            sensors = HybridControllerObservation.from_sim(self.sim, self.fly.name)
            command = self.controller.step(drive, sensors)
            apply_locomotion_action(self.sim, self.fly.name, command)
            self.sim.step()
        # Refresh kinematic fields after integration (mj_step leaves some pre-step).
        mujoco.mj_forward(self.sim.mj_model, self.sim.mj_data)
        if (not np.isfinite(self.sim.mj_data.qpos).all()
                or not np.isfinite(self.sim.mj_data.qvel).all()
                or self.sim.mj_data.time < initial_time):
            raise FloatingPointError("MuJoCo state invalid; inspect simulation parameters")
        self.elapsed_steps += 1
        velocity = (self.position - previous_position) / self.config.control_dt
        # A 100 ms velocity filter avoids rewarding individual foot-strike spikes.
        alpha = 1.0 - np.exp(-self.config.control_dt / 0.1)
        self.filtered_velocity += alpha * (velocity - self.filtered_velocity)
        error = float(np.linalg.norm(self.filtered_velocity[:2] - self.target_velocity))
        heading_error = float(abs(np.arctan2(
            np.sin(self.target_heading - self.heading),
            np.cos(self.target_heading - self.heading))))
        action_change = float(np.mean((action - self.last_action) ** 2))
        upright = float(self.rotation[2, 2])
        fallen = bool(upright < 0.3 or self.position[2] < 0.3)
        reward = self.config.control_dt * (
            np.exp(-(error / 5.0) ** 2)
            + 0.1 * np.cos(heading_error) + 0.05 * upright
            - 0.02 * action_change)
        if fallen:
            reward -= 1.0
        self.last_action = action.copy()
        self.episode_return += float(reward)
        self.tracking_errors.append(error)
        self.heading_errors.append(heading_error)
        self.action_changes.append(action_change)
        truncated = self.elapsed_steps >= self.max_steps
        self.done = fallen or truncated
        info = {
            "velocity_error_mm_s": error,
            "heading_error_deg": float(np.rad2deg(heading_error)),
            "upright": upright, "fallen": fallen,
            "elapsed_seconds": self.elapsed_steps * self.config.control_dt,
            "x_mm": float(self.position[0]), "y_mm": float(self.position[1]),
            "z_mm": float(self.position[2]),
        }
        if self.done:
            info["metrics"] = {
                "return": self.episode_return,
                "mean_velocity_error_mm_s": float(np.mean(self.tracking_errors)),
                "mean_heading_error_deg": float(np.rad2deg(np.mean(self.heading_errors))),
                "mean_action_change_sq": float(np.mean(self.action_changes)),
                "distance_mm": float(np.linalg.norm(
                    self.position[:2] - self.start_position[:2])),
                "fallen": fallen,
                "duration_s": info["elapsed_seconds"],
                "target_speed_mm_s": self.target_speed,
            }
        return self._observation(), float(reward), fallen, bool(truncated), info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.sim.mj_model, height=480, width=640)
        if self.camera_follow_body:
            self.camera.lookat[:] = self.position
        self.renderer.update_scene(self.sim.mj_data, camera=self.camera)
        return self.renderer.render().copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.sim.close()
