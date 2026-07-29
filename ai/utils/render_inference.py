import matplotlib
matplotlib.use("Agg") 

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import numpy as np
import torch
import json

from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import NoopLogger

import bsc_core
from gymnasium.spaces import Box, Discrete

from ai.model import RLTwoLayerCfcModel
from ai.environment import MazeEnvironment
from ai.training import load_model

def render_agent_trajectory(checkpoint_path: str, chromosome_json_path: str, output_image_path: str = "agent_trajectory.png", idx: int = 10, maze_size: int = 8):
    device = "cpu"
    print(f"[ИНФЕРЕНС] Используем устройство: {device}")
    
    config = (
        PPOConfig()
        .experimental(_enable_new_api_stack=False)
        .experimental(_disable_preprocessor_api=True)

        .environment(
            env=MazeEnvironment,
            env_config={
                "maze_size": MAZE_SIZE,
                "diamond_radius": 5,
                "maze_manager": None
            },
            normalize_actions=True
        )

        .framework("torch")

		.rollouts(
			num_rollout_workers=1,
			rollout_fragment_length='auto',
			batch_mode="complete_episodes"
		)


        .training(
            lr=5e-5,
            gamma=0.97,
            train_batch_size=4096,
            num_sgd_iter=3,
            entropy_coeff=0.15,
            sgd_minibatch_size=256,
            vf_clip_param=0.5,
            vf_loss_coeff=0.05,
            clip_param=0.2,
            model={
                "custom_model": RLTwoLayerCfcModel,
                "max_seq_len": 128,
                "use_lstm": False,
                "custom_model_config": {
                	"device": device
                }
            },
            grad_clip=1.0
        )

        .resources(
            num_gpus=0,
            num_cpus_for_local_worker=1,
            num_cpus_per_worker=1
        )

        .debugging(log_level="INFO")
    )

    algo = config.build(logger_creator=lambda conf: NoopLogger(config=conf, logdir="/tmp", trial=None))
    load_model(algo, checkpoint_path)

    with open(chromosome_json_path, 'r') as file:
        population_data = json.load(file)

    target_data = population_data[idx]
    chromosome = target_data["chromosome"]
    maze_size = target_data["maze_size"]

    env_config = {"maze_size": maze_size, "diamond_radius": 5, "maze_manager": None}
    env = MazeEnvironment(config=env_config)
    env.without_manager = True
    env.current_maze = bsc_core.generate_with_genome(chromosome, maze_size) 

    obs, info = env.reset()

    path_x = [env.pos[0]]
    path_y = [env.pos[1]]

    print(f"[СИМУЛЯЦИЯ] Запуск CfC-инференса...")
    total_reward = 0.0
    success = False

    hidden_state = algo.get_policy().model.get_initial_state()

    for step in range(512):
        action, hidden_state, _ = algo.compute_single_action(observation=obs, state=hidden_state, explore=False)

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        path_x.append(env.pos[0])
        path_y.append(env.pos[1])


        if done:
            success = True
            print(f"🎉 CfC-Агент триумфально ДОШЕЛ до финиша! Шагов: {env.steps_cnt}, Награда: {total_reward:.2f}")
            break
        if truncated:
            print(f"🛑 Тайм-аут на инференсе! Шагов: {env.steps_cnt}, Награда: {total_reward:.2f}")
            break

    print("[РЕНДЕР] Отрисовка траектории на матрице...")
    matrix = np.array(env.cur_maze_mat)
    fig, ax = plt.subplots(figsize=(12, 12), dpi=150)
    
    maze_cmap = ListedColormap(['#1a1a1a', '#ffffff', '#2ecc71']) 

    plt.imshow(matrix, cmap=maze_cmap, origin="upper")

    visited_map = np.array(env.visited)
    masked_visited = np.ma.masked_where(visited_map == 0, visited_map)
    ax.imshow(masked_visited, cmap=plt.cm.autumn, origin="upper", alpha=0.35)

    st_x, st_y = env.start
    fn_x, fn_y = env.finish

    ax.plot(path_x, path_y, color="#3498db", linewidth=2.5, alpha=0.9, label="Путь агента (CfC Memory)", zorder=3)
    ax.scatter(path_x[-1], path_y[-1], color="cyan", s=100, marker="o", edgecolors="black", label="Конечная точка", zorder=4)

    ax.scatter(st_x, st_y, color="green", s=90, marker='s', alpha=0.9, edgecolors="black", label=f"Старт ({st_x}, {st_y})", zorder=5)
    ax.scatter(fn_x, fn_y, color="red", s=100, marker='x', alpha=0.9, linewidths=3, label=f"Финиш ({fn_x}, {fn_y})", zorder=5)

    first_dot = chromosome.find('.')
    second_dot = chromosome.find('.', first_dot + 1)
    gen_y = int(chromosome[:first_dot])
    gen_x = int(chromosome[first_dot + 1:second_dot])
    ax.scatter(gen_x, gen_y, color="blue", s=70, marker="o", alpha=0.6, edgecolors="black", label="Точка зарождения ГА", zorder=4)

    status_str = "ДОШЕЛ ДО ЦЕЛИ" if success else "ЗАСТРЯЛ В КОРИДОРЕ"
    info_text = (
        f"Статус: {status_str}\n"
        f"Всего шагов: {env.steps_cnt}\n"
        f"Набрано наград: {total_reward:.2f}"
    )
    fig.text(0.14, 0.82, info_text, fontsize=11, fontweight='bold', color='black',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.5'), zorder=6)

    ax.set_title(f"Визуализация траектории инференса CfC-агента (Размер {env.maze_size}x{env.maze_size})", fontsize=14, pad=15)
    ax.legend(loc="upper right", fontsize=10)
    ax.axis("off")

    plt.savefig(output_image_path, bbox_inches="tight")
    plt.close()
    print(f"💾 График траектории успешно сохранен в: {output_image_path}")

if __name__ == "__main__":
    PATH_TO_WEIGHTS = "../models/model_gen_23.pth" 		
    PATH_TO_JSON = "../generations/generation_23.json" 	
    CHROMOSOME_IDX = 10 						
    MAZE_SIZE = 8							
    
    render_agent_trajectory(PATH_TO_WEIGHTS, PATH_TO_JSON, "agent_trajectory.png", CHROMOSOME_IDX, MAZE_SIZE)


