"""Maze topology, local sensing, return decisions and non-farmable rewards."""
from dataclasses import fields
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from gymnasium.utils.env_checker import check_env

from cyberfly.maze import MAZE_KINDS, RAY_ANGLES, MazeConfig, MazeLayout
from cyberfly.maze_brain import MAZE_FRAME_SIZE, ODOMETRY, WALLS, DopamineSystem, MazeBrain
from cyberfly.maze_env import MazeEnv, reward_components
from cyberfly.maze_learning import require_device


class MazeTopologyTests(unittest.TestCase):
    def test_all_generators_connected_with_a_deepest_reachable_single_goal(self):
        for kind in MAZE_KINDS:
            for seed in range(12):
                with self.subTest(kind=kind,seed=seed):
                    layout = MazeLayout(MazeConfig(kind=kind),np.random.default_rng(seed))
                    distances = layout.distances(0)
                    self.assertEqual(len(distances),9)
                    self.assertEqual(distances[layout.goal],max(distances.values()))
                    self.assertNotEqual(layout.goal,0)
                    self.assertEqual(layout.report()["food_count"],1)
                    self.assertGreater(layout.odor(layout.center(layout.goal)),0.98)
                    if kind == "looped":
                        self.assertGreaterEqual(len(layout.edges),layout.n)
                    else:
                        self.assertEqual(len(layout.edges),layout.n-1)

    def test_walls_block_vision_and_odor_uses_passages(self):
        cfg = MazeConfig(rows=2,cols=2,kind="corridor")
        maze = MazeLayout(cfg,np.random.default_rng(1))
        self.assertEqual(maze.goal,2)
        goal = maze.center(maze.goal)
        self.assertFalse(maze.visible([0,0],goal))
        self.assertTrue(maze.visible([8,8],goal))
        self.assertAlmostEqual(maze.ray_distances([0,0],[np.pi/2],20)[0],3.6)
        self.assertEqual(maze.odor([0,4]),0)
        # The geometrically farther point is closer by the traversable corridor.
        self.assertGreater(np.linalg.norm([8,0]-goal),np.linalg.norm(goal))
        self.assertGreater(maze.odor([8,0]),maze.odor([0,0]))
        self.assertTrue(np.isfinite(maze.odor_distance[maze.free]).all())

    def test_layout_reproducibility(self):
        cfg = MazeConfig()
        first = MazeLayout(cfg,np.random.default_rng(56))
        second = MazeLayout(cfg,np.random.default_rng(56))
        self.assertEqual(first.report(),second.report())
        np.testing.assert_array_equal(first.concentration,second.concentration)

    def test_invalid_numeric_and_geometry_configurations(self):
        for field in fields(MazeConfig):
            if field.name == "kind":
                continue
            for bad in (float('nan'),float('inf'),-float('inf'),True,'2'):
                with self.subTest(field=field.name,value=bad),self.assertRaises(ValueError):
                    MazeConfig(**{field.name:bad})
        for settings in ({"rows":1,"cols":1},{"rows":2.5},{"wall_mm":0.1},
                         {"cell_mm":3},{"odor_grid_mm":0.8},{"gamma":1},
                         {"history_frames":0},{"episode_seconds":0.03},
                         {"intake_reward":0},{"completion_reward":-1},{"kind":"unknown"}):
            with self.subTest(settings=settings),self.assertRaises(ValueError):
                MazeConfig(**settings)

    def test_mps_requirement_never_silently_falls_back(self):
        with patch.dict('os.environ',{'PYTORCH_ENABLE_MPS_FALLBACK':'0'}), \
                patch('torch.backends.mps.is_available',return_value=False):
            with self.assertRaisesRegex(RuntimeError,'No CPU fallback'):
                require_device('mps')
        with patch.dict('os.environ',{'PYTORCH_ENABLE_MPS_FALLBACK':'1'}):
            with self.assertRaisesRegex(RuntimeError,'Unset PYTORCH_ENABLE_MPS_FALLBACK'):
                require_device('mps')


class MazeBehaviorTests(unittest.TestCase):
    def sensory_frame(self, openings, velocity=(0,0)):
        frame = np.zeros(MAZE_FRAME_SIZE,dtype=np.float32)
        frame[57] = 0.45
        frame[48:50] = 0.2
        frame[WALLS] = 3.6/14
        for angle in openings:
            ray = np.argmin(abs(RAY_ANGLES-angle))
            frame[90+ray] = 1
        frame[ODOMETRY] = [velocity[0]/20,velocity[1]/20,0,1]
        return frame

    def test_dead_end_returns_along_its_own_memory(self):
        brain = MazeBrain(8)
        brain.act(self.sensory_frame([0]),0.02)
        np.testing.assert_array_equal(brain.target,[1,0])
        # Reach the next junction through locally integrated displacement.
        brain.act(self.sensory_frame([-np.pi],velocity=(8,0)),1.)
        self.assertTrue(brain.backtracking)
        self.assertEqual(brain.backtracks,1)
        np.testing.assert_array_equal(brain.target,[0,0])
        self.assertEqual(brain.mode,"backtrack")

    def test_neural_branch_action_changes_unexplored_choice(self):
        targets = []
        for bias in (-1,1):
            brain = MazeBrain(8)
            brain.branch_bias = bias
            brain.act(self.sensory_frame([-np.pi/2,np.pi/2]),0.02)
            targets.append(brain.target.tolist())
        self.assertEqual(targets,[[0,-1],[0,1]])

    def test_bilateral_odor_drives_junction_choice(self):
        targets = []
        for left,right in ((0.3,0.1),(0.1,0.3)):
            brain = MazeBrain(8)
            frame = self.sensory_frame([-np.pi/2,np.pi/2])
            frame[48:50] = [left,right]
            brain.act(frame,0.02)
            targets.append(brain.target.tolist())
        self.assertEqual(targets,[[0,1],[0,-1]])

    def test_odor_loops_do_not_generate_positive_net_shaping(self):
        cfg = MazeConfig()
        first = reward_components(cfg,0.2,0.8,0,False,False,False,np.zeros(4))
        second = reward_components(cfg,0.8,0.2,0,False,False,False,np.zeros(4))
        self.assertGreater(first['odor'],0)
        self.assertLess(second['odor'],0)
        self.assertLessEqual(first['odor']+cfg.gamma*second['odor'],0)
        final = reward_components(cfg,0.9,0,1,True,True,False,np.zeros(4))
        self.assertEqual(final['completion'],15)
        self.assertEqual(final['intake'],8)
        self.assertGreater(sum(final.values()),20)
        invalid = reward_components(cfg,0.9,0,1,True,True,True,np.zeros(4))
        self.assertEqual(invalid['completion'],0)
        self.assertEqual(invalid['intake'],0)
        truncation = reward_components(cfg,0.2,0.8,0,False,False,False,np.zeros(4))
        self.assertGreater(truncation['odor'],0)

    def test_dopamine_is_a_finite_prediction_error_not_an_extra_reward(self):
        model = DopamineSystem(0.999,0.5)
        frame = self.sensory_frame([0])
        before = model.observation().copy()
        after = model.update(frame,frame,8,False,0.02)
        self.assertGreater(after[0],before[0])
        self.assertGreater(model.error,0)
        model.update(frame,frame,-2,True,0.02)
        self.assertGreater(model.aversive,0)
        self.assertTrue(np.isfinite(model.observation()).all())
        model.reset()
        np.testing.assert_array_equal(model.observation(),np.zeros(3))

    def test_physical_sensor_contract_and_reproducibility(self):
        env = MazeEnv(MazeConfig(rows=2,cols=2,kind="corridor",episode_seconds=0.2))
        try:
            check_env(env,skip_render_check=True)
            a,info = env.reset(seed=7)
            positions = env.food_positions.copy()
            b,again = env.reset(seed=7)
            np.testing.assert_array_equal(a,b)
            np.testing.assert_array_equal(positions,env.food_positions)
            self.assertEqual(info,again)
            self.assertEqual(env.food_count,1)
            self.assertEqual(a.shape,(16,126))
            self.assertTrue(np.all(a[-1,:48] == 0),'Cannot see sugar through the separating wall')
            self.assertTrue(np.all(a[-1,48:50] > 0),'Paired odor reaches the entrance along the corridor')
            for invalid in ([0,0,0],[0,0,float('nan'),0]):
                with self.assertRaises(ValueError):
                    env.step(invalid)
        finally:
            env.close()

    def test_walls_create_actual_body_contacts_and_block_a_forced_drive(self):
        env = MazeEnv(MazeConfig(rows=2,cols=2,kind="corridor",episode_seconds=2))
        try:
            env.reset(seed=0)
            model,data = env.sim.mj_model,env.sim.mj_data
            joint = int(np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)[0])
            adr = model.jnt_qposadr[joint]
            data.qpos[adr:adr+2] += [-env.position[0],3.6-env.position[1]]
            mujoco.mj_forward(model,data)
            self.assertTrue(any(c.geom1 in env.wall_geoms or c.geom2 in env.wall_geoms for c in data.contact))
            env.reset(seed=0)
            data.qpos[adr:adr+2] += [-env.position[0],1.2-env.position[1]]
            data.qpos[adr+3:adr+7] = [np.cos(np.pi/4),0,0,np.sin(np.pi/4)]
            mujoco.mj_forward(model,data)
            # Intentionally disable navigation to test the wall, not avoidance logic.
            env.brain.act = lambda frame,dt:np.array([0.65,0.65,-1.])
            for _ in range(env.max_steps):
                _,_,terminated,truncated,_ = env.step(np.zeros(4))
                if terminated or truncated:
                    break
            self.assertGreater(env.wall_contacts,0)
            self.assertLess(env.position[1],4.4)
            env.reset(seed=0)
            env._is_escaped = lambda:True
            self.assertEqual(env._ingest(1.,0.),(0.,False))
            self.assertEqual(env.consumed,0.)
        finally:
            env.close()


if __name__ == '__main__':
    unittest.main()
