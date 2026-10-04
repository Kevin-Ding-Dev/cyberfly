"""Physical single-source maze foraging with local senses and auditable rewards."""
from collections import deque

import gymnasium as gym
import mujoco
import numpy as np

from .foraging import ForageConfig, ForagingEnv
from .maze import MazeConfig, MazeLayout, RAY_ANGLES
from .maze_brain import (DOPAMINE, MAZE_FRAME_SIZE, MEMORY, DopamineSystem, MazeBrain)


def reward_components(config, previous_odor, odor, intake, finished, terminal, failed, action, dt=0.02):
    """Discounted sensory potential plus actual intake and a one-off completion reward."""
    phi_next = 0. if terminal else config.odor_reward*odor
    return {"odor":config.gamma*phi_next-config.odor_reward*previous_odor,
            "intake":float(config.intake_reward*intake) if not failed else 0.,
            "completion":config.completion_reward if finished and not failed else 0.,
            "time":-config.time_cost*dt,"effort":-0.01*dt*float(np.mean(np.asarray(action)**2)),
            "failure":-2. if failed else 0.}


class MazeEnv(ForagingEnv):
    def __init__(self, config=None, render_mode=None):
        self.maze_config = config or MazeConfig()
        cfg = self.maze_config
        self.layout = MazeLayout(cfg,np.random.default_rng(0))
        self.dopamine = DopamineSystem(cfg.gamma,cfg.dopamine_tau_seconds)
        self.odor_previous = 0.0
        self.maze_active = False
        super().__init__(ForageConfig(episode_seconds=cfg.episode_seconds,food_min=1,food_max=1,
                         food_profile="sucrose-droplet",food_radius=0.4,droplet_volume_ul=0.1,
                         arena_radius=cfg.cell_mm*max(cfg.rows,cfg.cols)*2,
                         visual_range=5.,sensor_noise=0.,volatile_food_odor=False,
                         residual_scale=cfg.residual_scale,mouth_residual_scale=2.),render_mode)
        self.brain = MazeBrain(cfg.cell_mm)
        self.action_space = gym.spaces.Box(-1,1,(4,),dtype=np.float32)
        self.observation_space = gym.spaces.Box(-10,10,(cfg.history_frames,MAZE_FRAME_SIZE),dtype=np.float32)
        self.history = deque(maxlen=cfg.history_frames)
        self.wall_mocap = [self.sim.mj_model.body(f"maze_wall_{i}").mocapid[0]
                           for i in range(len(self.layout.wall_slots()))]
        self.wall_geoms = [self.sim.mj_model.geom(f"maze_wall_geom_{i}").id
                          for i in range(len(self.wall_mocap))]
        self.camera.lookat[:] = [(cfg.cols-1)*cfg.cell_mm/2,(cfg.rows-1)*cfg.cell_mm/2,0]
        self.camera.distance = max(24,cfg.cell_mm*max(cfg.rows,cfg.cols)*1.8)
        self.camera.azimuth = 90
        self.camera.elevation = -75

    def _configure_fly(self):
        super()._configure_fly()
        # FlyGym disables automatic collision on body meshes and explicitly pairs
        # them with the floor. A separate mask enables walls without self-contact.
        for geoms in self.fly.bodyseg_to_mjcfgeom.values():
            for geom in geoms:
                geom.conaffinity = 2

    def _configure_world(self, world):
        super()._configure_world(world)
        h = self.maze_config.wall_height_mm/2
        for i,(_,box) in enumerate(self.layout.wall_slots()):
            x,y,hx,hy = box
            body = world.mjcf_root.worldbody.add_body(name=f"maze_wall_{i}",mocap=True,pos=[x,y,h])
            body.add_geom(name=f"maze_wall_geom_{i}",type=mujoco.mjtGeom.mjGEOM_BOX,
                          size=[hx,hy,h],rgba=[0.28,0.36,0.42,1],contype=2,conaffinity=0,
                          friction=[1,0.005,0.0001],group=0)

    def _sensors(self):
        base = super()._sensors()
        if not self.maze_active:
            return np.concatenate((base,np.zeros(MAZE_FRAME_SIZE-90))).astype(np.float32)
        alive = self.food_remaining[0] > 1e-8
        for side,site in enumerate(self.eye_sites):
            if not alive or not self.layout.visible(self.sim.mj_data.site_xpos[site],self.food_positions[0]):
                base[side*24:(side+1)*24] = 0
        for side,site in enumerate(self.antenna_sites):
            base[48+side] = self.layout.odor(self.sim.mj_data.site_xpos[site]) if alive else 0.
        rays = self.layout.ray_distances(self.position,RAY_ANGLES+self.heading,self.maze_config.cell_mm*1.75)
        if self.trajectory:
            previous = np.array([self.trajectory[-1]['x_mm'],self.trajectory[-1]['y_mm'],self.position[2]])
            world_velocity = (self.position-previous)/self.config.control_dt
            c,s = np.cos(self.heading),np.sin(self.heading)
            velocity = np.array([c*world_velocity[0]+s*world_velocity[1],
                                 -s*world_velocity[0]+c*world_velocity[1],0.])
        else:
            velocity = np.zeros(3)
        odor = float(np.mean(base[48:50]))
        extras = np.concatenate((rays/(self.maze_config.cell_mm*1.75),
                    [velocity[0]/20,velocity[1]/20,np.sin(self.heading),np.cos(self.heading)],
                    [np.clip((odor-self.odor_previous)*50,-1,1)],self.dopamine.observation(),self.brain.memory()))
        frame = np.concatenate((base,extras)).astype(np.float32)
        if frame.shape != (MAZE_FRAME_SIZE,) or not np.isfinite(frame).all():
            raise FloatingPointError("Invalid maze sensory frame")
        return np.clip(frame,-10,10)

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError("Maze food is fixed by deepest reachable cell; layout overrides are unsupported")
        self.maze_active = False
        super().reset(seed=seed,options={"food_positions":[[0,0]]})
        self.layout = MazeLayout(self.maze_config,self.np_random)
        self.dopamine.reset()
        self.odor_previous = 0.
        self.food_positions[0,:2] = self.layout.center(self.layout.goal)
        self._sync_food()
        h = self.maze_config.wall_height_mm/2
        for i,(edge,box) in enumerate(self.layout.wall_slots()):
            active = edge is None or edge not in self.layout.edges
            self.sim.mj_data.mocap_pos[self.wall_mocap[i]] = [*box[:2],h] if active else [0,0,-100]
            self.sim.mj_model.geom_rgba[self.wall_geoms[i],3] = float(active)
        mujoco.mj_forward(self.sim.mj_model,self.sim.mj_data)
        self.maze_active = True
        self._motion_origin = self.position[:2].copy()
        self.frame = self._sensors()
        self.odor_previous = float(np.mean(self.frame[48:50]))
        self.frame[118] = 0
        self.history.clear()
        self.history.extend([self.frame.copy() for _ in range(self.maze_config.history_frames)])
        self.reward_totals = dict(odor=0.,intake=0.,completion=0.,time=0.,effort=0.,failure=0.)
        self.actual_visited = {0}
        self.wall_contacts = 0
        self.trajectory[0].update(odor=self.odor_previous,dopamine_appetitive=0.,dopamine_aversive=0.,
                                  dopamine_td_error=0.,backtracking=0,branch_action=0.)
        return np.stack(self.history),{"maze":self.layout.report()}

    def _is_escaped(self):
        # Keep this a planar task even if a policy attempts climbing or tunneling.
        return bool(self.layout.cell_at(self.position[:2]) is None
                    or self.position[2] > self.maze_config.wall_height_mm
                    or not self.layout.visible(self._motion_origin,self.position[:2]))

    def _ingest(self, mouth_command, speed):
        if self._is_escaped() or self.position[2] < 0.3 or self.rotation[2,2] < 0.3:
            self.sip_elapsed,self.sip_target = 0.,None
            return 0.,False
        return super()._ingest(mouth_command,speed)

    def step(self, action):
        action = np.asarray(action,dtype=np.float32)
        if action.shape != (4,) or not np.isfinite(action).all():
            raise ValueError("Maze policy requires four finite actions")
        action = np.clip(action,-1,1)
        before = self.frame.copy()
        self._motion_origin = self.position[:2].copy()
        old_consumed,old_return = self.consumed,self.episode_return
        self.brain.branch_bias = float(action[3])
        _,_,terminated,truncated,info = super().step(action[:3])
        cfg,dt = self.maze_config,self.config.control_dt
        odor = float(np.mean(self.frame[48:50]))
        # Signed discounted potential: repeated sniffing/back-and-forth is not free reward.
        components = reward_components(cfg,self.odor_previous,odor,self.consumed-old_consumed,
                                       self.food_remaining[0] <= 1e-8,terminated,
                                       info["fallen"] or info["escaped"],action,dt)
        reward = float(sum(components.values()))
        if not np.isfinite(reward):
            raise FloatingPointError("Non-finite maze reward; reduce reward coefficients")
        self.episode_return = old_return+reward
        self.dopamine.update(before,self.frame,reward,terminated,dt)
        self.frame[DOPAMINE] = self.dopamine.observation()
        self.frame[MEMORY] = self.brain.memory()
        self.history[-1] = self.frame.copy()
        self.odor_previous = odor
        for key,value in components.items():
            self.reward_totals[key] += value
        cell = self.layout.cell_at(self.position[:2])
        if cell is not None:
            self.actual_visited.add(cell)
        wall_ids = set(self.wall_geoms)
        self.wall_contacts += int(any(c.geom1 in wall_ids or c.geom2 in wall_ids for c in self.sim.mj_data.contact))
        details = {"odor":odor,"dopamine_appetitive":self.dopamine.appetitive,
                   "dopamine_aversive":self.dopamine.aversive,"dopamine_td_error":self.dopamine.error,
                   "backtracking":int(self.brain.backtracking),"branch_action":float(action[3])}
        self.trajectory[-1].update(details)
        info.update(details,reward_components=components)
        if self.done:
            info["metrics"]["return"] = self.episode_return
            if info["fallen"] or info["escaped"]:
                info["metrics"]["ate_all"] = False
            info["metrics"].pop("task_score",None)
            info["metrics"].update(maze_kind=self.layout.kind,goal_depth_edges=self.layout.depth[self.layout.goal],
                                   visited_cells=len(self.actual_visited),maze_cells=self.layout.n,
                                   backtracks=self.brain.backtracks,wall_contact_steps=self.wall_contacts,
                                   reward_components=self.reward_totals.copy())
        return np.stack(self.history),reward,terminated,truncated,info
