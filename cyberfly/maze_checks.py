"""Public maze checks, including a real turn-and-ingest physical rollout."""
import numpy as np
from gymnasium.utils.env_checker import check_env

from .maze import MAZE_KINDS, MazeConfig, MazeLayout
from .maze_env import MazeEnv


def run_checks():
    for kind in MAZE_KINDS:
        for seed in range(10):
            maze = MazeLayout(MazeConfig(kind=kind),np.random.default_rng(seed))
            assert len(maze.depth) == maze.n
            assert maze.depth[maze.goal] == max(maze.depth.values())
            assert maze.report()['food_count'] == 1
    print('PASS: all maze generators connected; exactly one goal at maximum graph depth',flush=True)
    layout = MazeLayout(MazeConfig(rows=2,cols=2,kind='corridor'),np.random.default_rng(0))
    assert not layout.visible([0,0],layout.center(layout.goal))
    assert layout.odor([0,4]) == 0
    assert layout.odor([8,0]) > layout.odor([0,0])
    print('PASS: walls occlude vision and odor follows traversable space',flush=True)
    short = MazeEnv(MazeConfig(rows=2,cols=2,kind='corridor',episode_seconds=0.2))
    try:
        check_env(short,skip_render_check=True)
    finally:
        short.close()
    print('PASS: Gymnasium contract, finite observations/actions and seeded resets',flush=True)
    env = MazeEnv(MazeConfig(rows=2,cols=2,kind='corridor',episode_seconds=12))
    try:
        env.reset(seed=0)
        total = 0.
        for _ in range(env.max_steps):
            _,reward,terminated,truncated,info = env.step(np.zeros(4))
            total += reward
            if terminated or truncated:
                break
        metrics = info['metrics']
        assert metrics['ate_all'] and not metrics['fallen'] and not metrics['escaped']
        assert metrics['visited_cells'] == 4
        assert np.isclose(sum(event['amount'] for event in env.events),1)
        assert np.isclose(total,metrics['return'])
        assert metrics['reward_components']['completion'] == env.maze_config.completion_reward
        assert np.isclose(metrics['reward_components']['intake'],env.maze_config.intake_reward)
        assert any(row['dopamine_appetitive'] > 0 for row in env.trajectory)
        assert env.food_remaining[0] == 0
        try:
            env.step(np.zeros(4))
        except RuntimeError:
            pass
        else:
            raise AssertionError('A completed goal cannot pay twice')
        print('PASS: physical U-maze traversal, real mouth contact, full intake, dopamine and one-off completion reward',flush=True)
    finally:
        env.close()
    retreat = MazeEnv(MazeConfig(kind='branching',episode_seconds=25))
    try:
        retreat.reset(seed=0)
        # A persistent left preference explores the non-food dead end first.
        for _ in range(retreat.max_steps):
            _,_,terminated,truncated,info = retreat.step(np.array([0.,0.,0.,1.]))
            if terminated or truncated:
                break
        metrics = info['metrics']
        assert metrics['backtracks'] >= 1
        assert metrics['ate_all'] and not metrics['fallen'] and not metrics['escaped']
        print('PASS: exploratory dead-end entry, physical retreat and subsequent full ingestion',flush=True)
    finally:
        retreat.close()
