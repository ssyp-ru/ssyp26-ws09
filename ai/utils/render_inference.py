import matplotlib

matplotlib.use("Agg") 

import matplotlib.pyplot as plt
import numpy as np
import torch
import json

from gymnasium.spaces import Box, Discrete

from ai.model import RLTwoLayerCfcModel
from ai.environment import MazeEnvironment


def render_agent_trajectory(checkpoint_path: str, chromosome_json_path: str, output_image_path: str = "agent_trajectory.png", idx: int = 12):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ИНФЕРЕНС] Используем устройство: {device}")

    obs_space = Box(low=-1.0, high=2.0, shape=(61,), dtype=np.float32)
    action_space = Discrete(4)
    num_outputs = 4
    model_config = {}

    model = RLTwoLayerCfcModel(obs_space, action_space, num_outputs, model_config, "inference_model")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if isinstance(checkpoint, dict) and "worker" in checkpoint and "policy_map" in checkpoint["worker"]:
        state_dict = checkpoint["worker"]["policy_map"]["default_policy"]["model"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    with open(chromosome_json_path, 'r') as file:
        population_data = json.load(file)

    target_data = population_data[idx]
    chromosome = target_data["chromosome"]

    env_config = {"maze_size": 128, "diamond_radius": 5, "maze_manager": None}
    env = MazeEnvironment(config=env_config)
    env.chromosome_apply_mode = True
    env.current_chromosome = chromosome

    obs, info = env.reset()

    path_x = [env.pos[0]]
    path_y = [env.pos[1]]


    print(f"[СИМУЛЯЦИЯ] Запуск прогона агента на карте...")
    total_reward = 0.0
    success = False

    hidden_state = [s.unsqueeze(0).to(device) for s in model.get_initial_state()]

    for step in range(4096):
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(device)
            input_dict = {"obs": obs_tensor}

            logits, next_state = model(input_dict, hidden_state, seq_lens=None)

            action = torch.argmax(logits, dim=-1).item()

            hidden_state = next_state

        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        path_x.append(env.pos[0])
        path_y.append(env.pos[1])

        if done:
            success = True
            print(f"🎉 Агент успешно ДОШЕЛ до финиша! Шагов сделано: {env.steps_cnt}, Награда: {total_reward:.2f}")
            break
        if truncated:
            print(f"🛑 Тайм-аут! Агент исчерпал лимит в {env.steps_cnt} шагов. Награда: {total_reward:.2f}")
            break

    print("[РЕНДЕР] Отрисовка траектории на матрице...")

    matrix = np.array(env.cur_maze_mat)

    plt.figure(figsize=(12, 12), dpi=150)

    plt.imshow(matrix, cmap=plt.cm.binary, origin="upper")

    visited_map = np.array(env.visited)
    masked_visited = np.ma.masked_where(visited_map == 0, visited_map)
    plt.imshow(masked_visited, cmap=plt.cm.autumn, origin="upper", alpha=0.4)

    plt.plot(path_x, path_y, color="#3498db", linewidth=1.5, alpha=0.8, label="Путь агента (CfC Memory)", zorder=3)

    plt.scatter(path_x[-1], path_y[-1], color="cyan", s=80, marker="o", edgecolors="black", label="Конечная точка агента", zorder=4)

    plt.scatter(env.start[0], env.start[1], color="green", s=45, marker='s', alpha=0.8, edgecolors="black", label=f"Старт ({env.start[0]}, {env.start[1]})", zorder=5)
    plt.scatter(env.finish[0], env.finish[1], color="red", s=45, marker='x', alpha=0.8, linewidths=2, label=f"Финиш ({env.finish[0]}, {env.finish[1]})", zorder=5)

    first_dot = chromosome.find('.')
    second_dot = chromosome.find('.', first_dot + 1)
    gen_y = int(chromosome[:first_dot])
    gen_x = int(chromosome[first_dot + 1:second_dot])

    plt.scatter(gen_x, gen_y, color="blue", s=50, marker="o", alpha=0.5, edgecolors="black", label="Точка зарождения ГА", zorder=4)

    status_str = "ДОШЕЛ ДО ЦЕЛИ" if success else "ЗАСТРЯЛ В КОРИДОРЕ"
    info_text = (
        f"Статус: {status_str}\n"
        f"Всего шагов: {env.steps_cnt}\n"
        f"Набрано наград: {total_reward:.2f}"
    )
    plt.text(3, 10, info_text, fontsize=11, fontweight='bold', color='black',
        bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', boxstyle='round,pad=0.5'), zorder=6)

    plt.title(f"Визуализация траектории инференса CfC-агента (Размер {env.maze_size}x{env.maze_size})", fontsize=14, pad=15)
    plt.legend(loc="upper right", fontsize=10)
    plt.axis("off")

    plt.savefig(output_image_path, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    PATH_TO_WEIGHTS = "../ai/models/model_gen_{поколение}.pth" 		# Путь к модели для инференса
    PATH_TO_JSON = "../ai/generations/generation_{поколение}.json" 	# Путь к .json файлу с хромосомами лабиринта.
    CHROMOSOME_IDX = 12 						# Номер хромосомы в файле.
    
    render_agent_trajectory(PATH_TO_WEIGHTS, PATH_TO_JSON, "agent_trajectory.png", CHROMOSOME_IDX)

