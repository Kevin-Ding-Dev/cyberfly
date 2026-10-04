"""Connected grid mazes, occlusion and a wall-respecting odor distance field."""
from collections import deque
from dataclasses import asdict, dataclass
import heapq
import math

import numpy as np

from .config_validation import require_time_multiple, validate_numbers

MAZE_KINDS = ("corridor", "branching", "looped")
RAY_ANGLES = np.linspace(-np.pi, np.pi, 24, endpoint=False)


@dataclass(frozen=True)
class MazeConfig:
    rows: int = 3
    cols: int = 3
    kind: str = "mixed"
    cell_mm: float = 8.0
    wall_mm: float = 0.8
    wall_height_mm: float = 3.0
    episode_seconds: float = 90.0
    odor_decay_mm: float = 32.0
    odor_grid_mm: float = 0.4
    history_frames: int = 16
    residual_scale: float = 0.35
    gamma: float = 0.999
    odor_reward: float = 3.0
    intake_reward: float = 8.0
    completion_reward: float = 15.0
    time_cost: float = 0.02
    dopamine_tau_seconds: float = 0.5
    version: int = 1

    def __post_init__(self):
        validate_numbers(self, integers=("rows", "cols", "history_frames", "version"), exclude=("kind",))
        if self.version != 1 or self.kind not in (*MAZE_KINDS, "mixed"):
            raise ValueError("Unsupported maze configuration")
        if not (1 <= self.rows <= 6 and 1 <= self.cols <= 6 and self.rows*self.cols >= 2):
            raise ValueError("Use 2–36 cells, with rows and columns between 1 and 6")
        if not (7 <= self.cell_mm <= 16 and 0.4 <= self.wall_mm <= 1.5
                and 2 <= self.wall_height_mm <= 6):
            raise ValueError("Maze dimensions must allow the fly to walk and turn")
        if not (0.2 <= self.odor_grid_mm <= self.wall_mm/2):
            raise ValueError("Odor grid must resolve walls: 0.2 <= grid <= wall/2 mm")
        require_time_multiple(self.cell_mm, self.odor_grid_mm)
        require_time_multiple(self.episode_seconds, 0.02)
        if not (4 <= self.history_frames <= 64 and 0 <= self.residual_scale <= 0.5):
            raise ValueError("Invalid history or motor correction range")
        if not (0 < self.gamma < 1 and self.odor_decay_mm > 0 and self.dopamine_tau_seconds > 0):
            raise ValueError("Invalid discount or sensory dynamics")
        if min(self.odor_reward, self.intake_reward, self.completion_reward, self.time_cost) < 0:
            raise ValueError("Reward weights must be nonnegative")
        if self.intake_reward <= 0 or self.completion_reward <= 0:
            raise ValueError("Both intake and completion must be rewarded")

    def to_dict(self):
        return asdict(self)


class MazeLayout:
    def __init__(self, config, rng):
        self.config = config
        self.kind = str(rng.choice(MAZE_KINDS)) if config.kind == "mixed" else config.kind
        self.n = config.rows*config.cols
        self.edges = set()
        if self.kind == "corridor":
            order = [r*config.cols+c for r in range(config.rows)
                     for c in (range(config.cols) if r%2 == 0 else range(config.cols-1, -1, -1))]
            self.edges = {self.edge(a, b) for a, b in zip(order, order[1:])}
        else:
            visited, stack = {0}, [0]
            while stack:
                current = stack[-1]
                choices = [n for n in self.neighbors(current) if n not in visited]
                if not choices:
                    stack.pop()
                    continue
                following = int(rng.choice(choices))
                self.edges.add(self.edge(current, following))
                visited.add(following)
                stack.append(following)
            if self.kind == "looped":
                closed = sorted({self.edge(a, b) for a in range(self.n) for b in self.neighbors(a)}-self.edges)
                if closed:
                    for index in rng.choice(len(closed), size=max(1, len(closed)//3), replace=False):
                        self.edges.add(closed[index])
        self.depth = self.distances(0)
        if len(self.depth) != self.n:
            raise RuntimeError("Maze generator produced an unreachable cell")
        deepest = [cell for cell, depth in self.depth.items() if depth == max(self.depth.values())]
        self.goal = int(rng.choice(deepest))
        self.walls = np.array([box for edge, box in self.wall_slots() if edge is None or edge not in self.edges])
        self._build_odor_field()

    @staticmethod
    def edge(a, b):
        return min(a, b), max(a, b)

    def neighbors(self, cell):
        r, c = divmod(cell, self.config.cols)
        return [nr*self.config.cols+nc for nr, nc in ((r-1,c), (r+1,c), (r,c-1), (r,c+1))
                if 0 <= nr < self.config.rows and 0 <= nc < self.config.cols]

    def distances(self, start):
        distances, queue = {start:0}, deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.neighbors(current):
                if self.edge(current, neighbor) in self.edges and neighbor not in distances:
                    distances[neighbor] = distances[current]+1
                    queue.append(neighbor)
        return distances

    def center(self, cell):
        r, c = divmod(cell, self.config.cols)
        return np.array([c*self.config.cell_mm, r*self.config.cell_mm])

    def cell_at(self, xy):
        c, r = np.floor(np.asarray(xy)/self.config.cell_mm+0.5).astype(int)
        if 0 <= r < self.config.rows and 0 <= c < self.config.cols:
            return int(r*self.config.cols+c)
        return None

    def wall_slots(self):
        """Every possible wall has one reusable MuJoCo mocap body."""
        cfg, slots = self.config, []
        s, t = cfg.cell_mm, cfg.wall_mm/2
        for r in range(cfg.rows):
            for c in range(cfg.cols+1):
                edge = None if c in (0, cfg.cols) else self.edge(r*cfg.cols+c-1, r*cfg.cols+c)
                slots.append((edge, [(c-0.5)*s, r*s, t, s/2+t]))
        for r in range(cfg.rows+1):
            for c in range(cfg.cols):
                edge = None if r in (0, cfg.rows) else self.edge((r-1)*cfg.cols+c, r*cfg.cols+c)
                slots.append((edge, [c*s, (r-0.5)*s, s/2+t, t]))
        return slots

    def ray_distances(self, origin, angles, limit):
        """Analytic horizontal depth rays against the same boxes as the physics walls."""
        origin = np.asarray(origin)[:2]
        directions = np.stack((np.cos(angles), np.sin(angles)), axis=-1)
        safe = np.where(abs(directions) < 1e-12, 1e-12, directions)
        lo, hi = self.walls[:,:2]-self.walls[:,2:], self.walls[:,:2]+self.walls[:,2:]
        a = (lo[None,:,:]-origin)/safe[:,None,:]
        b = (hi[None,:,:]-origin)/safe[:,None,:]
        enter = np.minimum(a,b).max(axis=-1)
        leave = np.maximum(a,b).min(axis=-1)
        distance = np.where((leave >= np.maximum(enter,0)), np.maximum(enter,0), limit)
        return np.minimum(distance.min(axis=1),limit)

    def visible(self, origin, target):
        delta = np.asarray(target)[:2]-np.asarray(origin)[:2]
        distance = np.linalg.norm(delta)
        return bool(self.ray_distances(origin,[math.atan2(delta[1],delta[0])],distance+1e-5)[0] >= distance)

    def _build_odor_field(self):
        cfg = self.config
        self.lower = np.array([-cfg.cell_mm/2]*2)
        spacing = cfg.odor_grid_mm
        nx, ny = round(cfg.cols*cfg.cell_mm/spacing), round(cfg.rows*cfg.cell_mm/spacing)
        x = self.lower[0]+(np.arange(nx)+0.5)*spacing
        y = self.lower[1]+(np.arange(ny)+0.5)*spacing
        xx, yy = np.meshgrid(x,y)
        self.free = np.ones((ny,nx),dtype=bool)
        for cx,cy,hx,hy in self.walls:
            self.free &= ~((abs(xx-cx) <= hx+1e-8)&(abs(yy-cy) <= hy+1e-8))
        gx, gy = np.floor((self.center(self.goal)-self.lower)/spacing).astype(int)
        distance = np.full((ny,nx),np.inf)
        distance[gy,gx] = 0
        queue = [(0.,gy,gx)]
        moves = [(dy,dx,math.hypot(dx,dy)*spacing) for dy in (-1,0,1) for dx in (-1,0,1) if dx or dy]
        while queue:
            value,r,c = heapq.heappop(queue)
            if value != distance[r,c]:
                continue
            for dr,dc,cost in moves:
                nr,nc = r+dr,c+dc
                if not (0 <= nr < ny and 0 <= nc < nx and self.free[nr,nc]):
                    continue
                if dr and dc and not (self.free[r,nc] and self.free[nr,c]):
                    continue
                trial = value+cost
                if trial < distance[nr,nc]:
                    distance[nr,nc] = trial
                    heapq.heappush(queue,(trial,nr,nc))
        if not np.isfinite(distance[self.free]).all():
            raise ValueError("Odor grid disconnected; adjust maze wall/grid dimensions")
        self.odor_distance = distance
        self.concentration = np.exp(-distance/cfg.odor_decay_mm)

    def odor(self, xy):
        point = np.asarray(xy)[:2]
        if any(np.all(abs(point-wall[:2]) <= wall[2:]) for wall in self.walls):
            return 0.0
        x,y = (point-self.lower)/self.config.odor_grid_mm-0.5
        c,r = math.floor(x),math.floor(y)
        if not (0 <= r < self.free.shape[0]-1 and 0 <= c < self.free.shape[1]-1):
            return 0.0
        fx,fy = x-c,y-r
        weights = np.array([[(1-fx)*(1-fy),fx*(1-fy)],[(1-fx)*fy,fx*fy]])
        mask = self.free[r:r+2,c:c+2]
        total = (weights*mask).sum()
        return float((self.concentration[r:r+2,c:c+2]*weights).sum()/total) if total > 0 else 0.0

    def report(self):
        return {"kind":self.kind,"rows":self.config.rows,"cols":self.config.cols,
                "start_cell":0,"goal_cell":self.goal,"goal_depth_edges":self.depth[self.goal],
                "goal_xy_mm":self.center(self.goal).tolist(),"food_count":1,
                "connected":len(self.depth)==self.n,"open_edges":[list(e) for e in sorted(self.edges)],
                "wall_boxes_xy_halfsize_mm":self.walls.tolist(),
                "odor_model":"exp(-free-space path distance / decay); volatile cue paired with sucrose"}
