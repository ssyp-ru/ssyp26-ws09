import ray
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.utils.typing import EpisodeType, PolicyID
from ray.rllib.evaluation import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.core.rl_module import RLModule
from ray.rllib.env.env_runner import EnvRunner
from ray.rllib import BaseEnv, Policy

import gymnasium as gym

import random

from typing import Union, Optional, Dict

import numpy as np

import bsc_core

@ray.remote
class GAMazeManager:
    def __init__(self, chromosomes_cnt: int, ms: int):
        self.population_size = chromosomes_cnt
        self.maze_size = ms

        self.population = self.generate_start_population()

        self.current_generation = 0

        self.rewards = [[] for _ in range(chromosomes_cnt)]
        self.ideal_rewards = [0.0 for _ in range(chromosomes_cnt)]
        self.fitness_scores = np.zeros(chromosomes_cnt)

        self.visits = []
        self.set_visits()

        self.current_chrom_idx = 0

    def set_visits(self):
        self.visits = list(range(self.population_size))

        random.shuffle(self.visits)

        self.reset_current_chrom_idx()

    def inc_current_chrom_idx(self):
        self.current_chrom_idx += 1

    def reset_current_chrom_idx(self):
        self.current_chrom_idx = 0

    def get_current_chrom_idx(self):
        return self.visits[min(self.current_chrom_idx, self.population_size - 1)]

    def get_current_chromosome(self):
        return self.population[self.get_current_chrom_idx()]

    def update_maze_size(self, new_size: int):
        print(f"[ГА-МЕНЕДЖЕР] Масштабируем популяцию под новый размер: {new_size}x{new_size}")
        self.maze_size = new_size

        self.population = [bsc_core.random_generate(self.maze_size) for _ in range(self.population_size)]

        self.fitness_scores = [0.0] * self.population_size
        self.visits = [0] * self.population_size
        self.current_chrom_idx = 0

    def get_current_chromosome_data(self):
        chrom_idx = self.get_current_chrom_idx()

        return chrom_idx, self.population[chrom_idx], self.maze_size

    def get_random_chromosome(self):
        chrom_idx = self.visits[self.current_chrom_idx]

        self.current_chrom_idx += 1

        if self.current_chrom_idx >= self.population_size:
            random.shuffle(self.visits)

            self.current_chrom_idx = 0

        return chrom_idx, self.population[chrom_idx]

    def generation_ready(self):
        return self.current_chrom_idx >= self.population_size

    def generate_start_population(self) -> list:
        return [bsc_core.random_generate(self.maze_size) for _ in range(self.population_size)]

    @staticmethod
    def evolution_step(population, fitness_scores) -> list:
        return bsc_core.select_and_crossover(population, fitness_scores)

    def new_generation(self):
        self.fitness_scores = [(self.ideal_rewards[k] - np.mean(self.rewards[k])) for k in range(self.population_size)]

        self.population = self.evolution_step(self.population, self.fitness_scores)

        self.rewards = [[] for _ in range(self.population_size)]
        self.current_generation += 1

        self.set_visits()

    def new_generation_with_ready_fitness_scores(self):
        self.population = self.evolution_step(self.population, self.fitness_scores)

        self.rewards = [[] for _ in range(self.population_size)]
        self.current_generation += 1

        self.set_visits()

    def record_reward(self, chrom_idx: int, reward: float):
        self.rewards[chrom_idx].append(reward)

    def record_ideal_reward(self, chrom_idx: int, reward: float):
        self.ideal_rewards[chrom_idx] = reward

    def evolve_if_ready(self, min_sample_per_chrome: int = 1) -> bool:
        if any(len(rewards) < min_sample_per_chrome for rewards in self.rewards):
            return False

        self.new_generation()
        return True

    def get_population(self):
        return self.population

    def set_population(self, _population):
        self.population = _population

    def get_fitness_scores(self):
        return self.fitness_scores

    def set_fitness_scores(self, fs):
        self.fitness_scores = fs


class CoEvolutionCallback(DefaultCallbacks):
    def __init__(self, ga_mng):
        super().__init__()

        self.ga_mng = ga_mng

    def on_episode_start(
            self,
            *,
            episode: Union[EpisodeType, Episode, EpisodeV2],
            worker: Optional["EnvRunner"] = None,
            env_runner: Optional["EnvRunner"] = None,
            base_env: Optional[BaseEnv] = None,
            env: Optional[gym.Env] = None,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            rl_module: Optional[RLModule] = None,
            env_index: int,
            **kwargs,
    ) -> None:

        if base_env is not None:
            sub_envs = base_env.get_sub_environments()

            if sub_envs:
                raw_env = sub_envs[kwargs.get("env_index", 0)].unwrapped

                chrome_idx, chromosome = ray.get(self.ga_mng.get_random_chromosome.remote())

                if hasattr(episode, "user_data"):
                    episode.user_data["current_chrom_idx"] = chrome_idx

                raw_env.current_chromosome = chromosome
                raw_env.current_chrom_idx = chrome_idx

    def on_episode_end(
            self,
            *,
            episode: Union[EpisodeType, Episode, EpisodeV2],
            worker: Optional["EnvRunner"] = None,
            env_runner: Optional["EnvRunner"] = None,
            base_env: Optional[BaseEnv] = None,
            env: Optional[gym.Env] = None,
            policies: Optional[Dict[PolicyID, Policy]] = None,
            rl_module: Optional[RLModule] = None,
            env_index: int,
            **kwargs,
    ) -> None:

        if hasattr(episode, "get_return"):
            agent_reward = episode.get_return()
        else:
            agent_reward = episode.total_reward

        if base_env is not None:
            sub_envs = base_env.get_sub_environments()

            if sub_envs:
                raw_env = sub_envs[kwargs.get("env_index", 0)].unwrapped

                if hasattr(episode, "user_data"):
                    chrom_idx = episode.user_data.get("current_chrom_idx")

                else:
                    chrom_idx = getattr(raw_env, "current_chrom_idx", None)

                if chrom_idx is not None:
                    self.ga_mng.record_reward.remote(chrom_idx, agent_reward)

                    ideal_reward = 2.0
                    self.ga_mng.record_ideal_reward.remote(chrom_idx, ideal_reward)
