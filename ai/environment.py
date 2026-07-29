import ray

import numpy as np

import gymnasium as gym
from gymnasium.spaces import Box, Discrete

import bsc_core
from oar_core import DiamondScanner, maze_bfs

WALL = -1.0
EMPTY = 0.0
SHADOW = 0.5
FINISH = 1.0

class MazeEnvironment(gym.Env):
    def __init__(self, config=None):
        super().__init__()

        self.maze_size = config.get("maze_size", 128)
        self.diamond_radius = config.get("diamond_radius")
        self.maze_manager = config.get("maze_manager")

        self.finish = None
        self.start = None

        self.current_maze = None
        self.cur_maze_mat = None
        self.pos = (0, 0)
        self.visited = np.zeros((self.maze_size, self.maze_size), dtype=np.int32)

        self.observation_space = Box(low=-1.0, high=1.0, shape=(11, 11), dtype=np.float32)
        self.action_space = Discrete(4)
        self.steps_cnt = 0

        self.observer = None
        self.dists = np.zeros((self.maze_size, self.maze_size))

        self.moves = ((-1, 0), (1, 0), (0, 1), (0, -1))

        self.total_reward = 0.0

        self.current_chromosome = None
        self.current_chrom_idx = None

        self.without_manager = False
        
        self.local_epoch_cache = None

        self.wall_reward = -0.1
        self.time_reward = (-1.0 / max(128, self.maze_size * 2))
        self.potential_reward_coef = 2.0
        self.visit_reward_coef = -0.05
        self.finish_reward = 2.0

    def reset(self, *, seed: int = None, options: dict = None) -> tuple:
        super().reset(seed=seed)
        self.total_reward = 0.0
        self.steps_cnt = 0

        if self.observer is None:
            self.observer = DiamondScanner(self.diamond_radius)

        if self.maze_manager is not None and not self.without_manager:
            chrom_idx, current_generation_ref = ray.get(self.maze_manager.get_random_maze_data.remote())
            self.current_chrom_idx = chrom_idx

            if self.local_epoch_cache is None or getattr(self, "last_fetched_ref", None) != current_generation_ref:
                if current_generation_ref is not None:
                    self.local_epoch_cache = ray.get(current_generation_ref)
                    self.last_fetched_ref = current_generation_ref
                else:
                    gen_ref = ray.get(self.maze_manager.get_maze_cache_ref.remote())
                    self.local_epoch_cache = ray.get(gen_ref)

            self.current_maze = self.local_epoch_cache[self.current_chrom_idx]
        elif not self.without_manager:
            if self.current_chromosome is None:
                self.current_chromosome = bsc_core.random_generate(self.maze_size)
            self.current_maze = bsc_core.generate_with_genome(self.current_chromosome, self.maze_size)


        self.cur_maze_mat = self.current_maze.get("mat")
        
        raw_finish = self.current_maze.get("finish")
        raw_start = self.current_maze.get("start")

        self.finish = (int(raw_finish[1]), int(raw_finish[0]))
        self.start = (int(raw_start[1]), int(raw_start[0]))
        self.pos = self.start

        self.visited = np.zeros((self.maze_size, self.maze_size), dtype=np.int32)

        if "dists" in self.current_maze:
            self.dists = self.current_maze["dists"]
        else:
            self.dists = maze_bfs(self.cur_maze_mat, self.finish[0], self.finish[1])


        self.steps_cnt = 0

        obs = self.observer.get_square_observe_with_shadow(self.cur_maze_mat, self.pos[0], self.pos[1])

        return obs, {"current_chrom_idx": getattr(self, "current_chrom_idx", -1)}

    def step(self, action):
        self.steps_cnt += 1
        
        if self.cur_maze_mat is None or len(self.cur_maze_mat) != self.maze_size:
            truncated = True
            
            grid_side = 2 * self.diamond_radius + 1
            stub_obs = np.full((grid_side, grid_side), -1.0, dtype=np.float32)
            
            self.time_reward = (-1.0 / max(128, self.maze_size * 2))

            return stub_obs, 0.0, False, truncated, {"current_chrom_idx": -1}

        nx_pos = (self.pos[0] + self.moves[action][0], self.pos[1] + self.moves[action][1])

        reward = 0.0
        done = False
        truncated = self.steps_cnt >= max(128, self.maze_size * self.maze_size / 4)

        if nx_pos[0] < 0 or nx_pos[0] >= self.maze_size or nx_pos[1] < 0 or nx_pos[1] >= self.maze_size:
            reward += self.wall_reward
            nx_pos = self.pos

        nx_cell = self.cur_maze_mat[nx_pos[1]][nx_pos[0]]

        if nx_cell == WALL:
            reward += self.wall_reward
            nx_pos = self.pos
            nx_cell = self.cur_maze_mat[nx_pos[1]][nx_pos[0]]

        reward += self.time_reward

        prev_dist = float(self.dists[self.pos[1], self.pos[0]])
        nx_dist = float(self.dists[nx_pos[1], nx_pos[0]])
        nx_vis_cnt = self.visited[nx_pos[1], nx_pos[0]]

        if nx_pos != self.pos:
            potential_delta = prev_dist - nx_dist
            normalized_potential = potential_delta / self.maze_size

            reward += normalized_potential * self.potential_reward_coef
            
            if nx_vis_cnt > 0:
                reward += self.visit_reward_coef * nx_vis_cnt

        self.pos = nx_pos
        self.visited[self.pos[1], self.pos[0]] += 1

        if nx_cell == FINISH or self.pos == self.finish:
            done = True
            reward += self.finish_reward

        self.total_reward += reward

        obs = self.observer.get_square_observe_with_shadow(self.cur_maze_mat, self.pos[0], self.pos[1])

        return obs, reward, done, truncated, {"current_chrom_idx": self.current_chrom_idx}

