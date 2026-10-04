"""Sensory exploration memory and a computational dopamine-like TD signal.

The navigator builds its own local grid memory from visual depth and idealized
proprioceptive odometry. It never receives maze edges, food coordinates or a
shortest route. This engineered scaffold is not a reconstructed fly connectome.
"""
import numpy as np
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from .brain import SensoryBrain
from .maze import RAY_ANGLES

WALLS = slice(90,114)
ODOMETRY = slice(114,118)  # body-frame planar velocity, sin/cos of yaw reference
ODOR_TREND = 118
DOPAMINE = slice(119,122)
MEMORY = slice(122,126)
MAZE_FRAME_SIZE = 126


def wrap(angle):
    return np.arctan2(np.sin(angle),np.cos(angle))


class DopamineSystem:
    """Per-episode linear TD predictor plus positive/negative decaying traces.

These are dimensionless proxies, not neurotransmitter concentrations. The
signal modulates observations; it does not add a second copy of task reward.
Long-term policy learning remains PPO, whose weights are checkpointed.
"""
    def __init__(self, gamma, tau):
        self.gamma,self.tau = gamma,tau
        self.reset()

    def reset(self):
        self.weights = np.zeros(5)
        self.eligibility = np.zeros(5)
        self.appetitive = self.aversive = self.error = 0.0

    @staticmethod
    def features(frame):
        return np.array([1.,float(np.mean(frame[48:50])),float(max(frame[:48])),
                         float(frame[56]),float(frame[89])])

    def update(self, previous, current, reward, terminal, dt):
        before,after = self.features(previous),self.features(current)
        prediction = float(self.weights@before)
        future = 0.0 if terminal else float(self.weights@after)
        self.error = float(np.clip(reward+self.gamma*future-prediction,-20,20))
        self.eligibility = self.gamma*0.9*self.eligibility+before
        self.weights = np.clip(self.weights+0.005*self.error*self.eligibility,-20,20)
        decay = np.exp(-dt/self.tau)
        self.appetitive = min(20.,decay*self.appetitive+max(0.,self.error))
        self.aversive = min(20.,decay*self.aversive+max(0.,-self.error))
        return self.observation()

    def observation(self):
        return np.tanh(np.array([self.appetitive,self.aversive,self.error])/5).astype(np.float32)


class MazeBrain:
    def __init__(self, cell_mm):
        self.cell_mm = cell_mm
        self.feeding_brain = SensoryBrain()
        self.reset()

    def reset(self):
        self.feeding_brain.reset()
        self.position = np.zeros(2)
        self.visited = set()
        self.stack = []
        self.target = None
        self.branch_bias = 0.0
        self.backtracking = False
        self.backtracks = 0
        self.mode = "odor_search"
        self.last_odor = 0.0
        self.odor_gradient = np.zeros(2)

    def memory(self):
        return np.array([float(self.backtracking),min(len(self.visited)/36,1),
                         self.branch_bias,float(self.target is not None)],dtype=np.float32)

    def act(self, frame, dt):
        vx,vy,sine,cosine = frame[ODOMETRY]
        heading = float(np.arctan2(sine,cosine))
        self.position += np.array([cosine*vx-sine*vy,sine*vx+cosine*vy])*20*dt
        # Bilateral odor contrast estimates the gradient perpendicular to the head.
        odor = float(np.mean(frame[48:50]))
        lateral = float(frame[48]-frame[49])/max(odor,1e-5)
        lateral_axis = np.array([-sine,cosine])
        self.odor_gradient = 0.98*self.odor_gradient+0.02*40*lateral*lateral_axis
        if vx > 0.03:
            forward = (odor-self.last_odor)/max(odor,1e-5)
            self.odor_gradient += 0.02*20*forward*np.array([cosine,sine])
        self.last_odor = odor
        if frame[56] > 0.5 or frame[57] >= 0.98 or max(frame[:48]) > 0.09:
            result = self.feeding_brain.act(frame[:90],dt)
            self.mode = self.feeding_brain.mode
            return result
        distances = frame[WALLS]*self.cell_mm*1.75
        if self.target is None or np.linalg.norm(self.target*self.cell_mm-self.position) < 0.7:
            node = tuple(np.rint(self.position/self.cell_mm).astype(int))
            self.visited.add(node)
            if node in self.stack:
                self.stack = self.stack[:self.stack.index(node)+1]
            else:
                self.stack.append(node)
            choices = []
            for direction in ((1,0),(0,1),(-1,0),(0,-1)):
                angle = float(np.arctan2(direction[1],direction[0]))
                ray = int(np.argmin(abs(wrap(RAY_ANGLES-(angle-heading)))))
                following = (node[0]+direction[0],node[1]+direction[1])
                if distances[ray] > self.cell_mm*0.7 and following not in self.visited:
                    score = float(self.odor_gradient@direction)
                    score += 0.7*self.branch_bias*np.sin(angle-heading)
                    score += 0.05*np.cos(angle-heading)
                    choices.append((score,following))
            if choices:
                self.target = np.array(max(choices,key=lambda item:item[0])[1])
                self.backtracking = False
            elif len(self.stack) > 1:
                self.target = np.array(self.stack[-2])
                self.backtracking = True
                self.backtracks += 1
            else:
                # All sensed branches exhausted: rescan instead of reading a route oracle.
                self.target = None
                self.mode = "rescan"
                return np.array([-0.35,0.35,-1.])
        delta = self.target*self.cell_mm-self.position
        error = float(wrap(np.arctan2(delta[1],delta[0])-heading))
        if self.backtracking and abs(error) > 2.2:
            # Back out of a narrow dead end instead of sweeping a full U-turn
            # into its side walls. Negative drive reverses the existing CPG.
            reverse_error = float(wrap(error-np.sign(error)*np.pi))
            turn = np.clip(1.1*reverse_error,-0.35,0.35)
            forward = -0.35*max(0.,np.cos(reverse_error))
            rear = float(distances[abs(abs(RAY_ANGLES)-np.pi) <= np.pi/12+1e-8].min())
            if rear < 2.0:
                forward = 0.
            self.mode = "backtrack"
            return np.array([forward-turn,forward+turn,-1.])
        turn = np.clip(1.25*error,-0.65,0.65)
        forward = 0.60*max(0.,np.cos(error))
        if abs(error) > 0.6:
            forward = 0.10
        front = float(distances[abs(RAY_ANGLES) <= np.pi/12+1e-8].min())
        if front < 2.0:
            forward = 0.0
            if abs(turn) < 0.15:
                turn = 0.55 if lateral >= 0 else -0.55
        self.mode = "backtrack" if self.backtracking else "odor_search"
        return np.array([forward-turn,forward+turn,-1.])


class MazeMemoryEncoder(BaseFeaturesExtractor):
    """Finite sensory window with dense layers supported by Apple MPS.

Longer exploration memory resides in the explicit navigator and dopamine state;
this encoder does not claim unbounded recurrent memory.
"""
    def __init__(self, observation_space, features_dim=256):
        super().__init__(observation_space,features_dim)
        history,channels = observation_space.shape
        self.frame = nn.Sequential(nn.Linear(channels,64),nn.ReLU())
        self.temporal = nn.Sequential(nn.Flatten(),nn.Linear(history*64,features_dim),nn.ReLU())

    def forward(self, observations):
        return self.temporal(self.frame(observations))
