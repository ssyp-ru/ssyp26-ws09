import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import ray
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.logger import NoopLogger

from ai.genetic_algorithm import GAMazeManager, CoEvolutionCallback
from ai.model import RLTwoLayerCfcModel
from ai.environment import MazeEnvironment

import torch

torch.set_num_threads(1)

from pathlib import Path


def save_pure_weights(algo, generation_num: int, checkpoint_dir_str: str):
    checkpoint_dir = Path(checkpoint_dir_str)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model_net = algo.get_policy("default_policy").model

    weight_file_path = checkpoint_dir / f"model_gen_{generation_num}.pth"
    torch.save(model_net.state_dict(), weight_file_path)
    print(f"[Успех] Чистые веса PyTorch сохранены в: {weight_file_path}")


def load_pure_weights_to_algorithm(algo, checkpoint_dir_str: str) -> int:
    checkpoint_dir = Path(checkpoint_dir_str)
    if not checkpoint_dir.exists():
        return 0

    weight_files = list(checkpoint_dir.glob("model_gen_*.pth"))
    if not weight_files:
        print("\n[Старт] Сохранённых весов PyTorch не найдено. Начинаем с нуля.\n")
        return 0

    def get_gen_num(path):
        try:
            return int(path.stem.split("_")[-1])
        except (ValueError, IndexError):
            return -1

    latest_file = max(weight_files, key=get_gen_num)
    latest_gen_num = get_gen_num(latest_file)

    if latest_gen_num >= 0:
        print(f"\n[Восстановление] Найден файл весов: {latest_file}")

        pure_state_dict = torch.load(latest_file, map_location="cpu")

        model_net = algo.get_policy("default_policy").model
        model_net.load_state_dict(pure_state_dict)

        algo.workers.local_worker().set_weights({"default_policy": algo.get_policy("default_policy").get_weights()})
        algo.workers.sync_weights()

        print(
            f"[Восстановление] Веса нейросети успешно восстановлены! Модель готова к поколению {latest_gen_num + 1}.\n")
        return latest_gen_num + 1

    return 0

def load_model(algo, checkpoint_path: str):
    print(f"\n[Восстановление] Файл весов: {checkpoint_path}")

    pure_state_dict = torch.load(checkpoint_path, map_location="cpu")

    model_net = algo.get_policy("default_policy").model
    model_net.load_state_dict(pure_state_dict)

    algo.workers.local_worker().set_weights({"default_policy": algo.get_policy("default_policy").get_weights()})
    algo.workers.sync_weights()

def load_last_epoch(generations_log: str, ga_manager) -> int:
    log_path = Path(generations_log)
    if log_path.is_dir() and any(log_path.iterdir()):
        json_files = [f for f in log_path.iterdir() if "generation_" in f.name and f.suffix == ".json"]
        if not json_files:
            return 0
        last_generation = max(int(fn.name[len("generation_"):-5]) for fn in json_files)
    else:
        return 0

    start_generation_path = log_path.joinpath(f"generation_{last_generation}.json")

    if start_generation_path.is_file():
        with open(start_generation_path, 'r') as file:
            start_population_data = json.load(file)
        start_population = [d["chromosome"] for d in start_population_data]
        start_fitness = [d["fitness_score"] for d in start_population_data]

        ray.get(ga_manager.set_population.remote(start_population))

        if hasattr(ga_manager, "set_fitness_scores"):
            ray.get(ga_manager.set_fitness_scores.remote(start_fitness))

        print(
            f"[ГА] Успешно загружено поколение {last_generation}. Проводим селекцию для перехода к {last_generation + 1}...")
        ray.get(ga_manager.new_generation_with_ready_fitness_scores.remote())

        return last_generation + 1

    return 0


def check_and_update_curriculum(algo, ga_manager, current_size, avg_ratio):
    SUCCESS_THRESHOLD = 0.75
    MAX_SIZE = 128
    MAZE_SIZE_STEP = 2
    
    if avg_ratio >= SUCCESS_THRESHOLD and current_size < MAX_SIZE:
        new_size = current_size + MAZE_SIZE_STEP
        if new_size > MAX_SIZE:
            new_size = MAX_SIZE
            
        print(f"\n🚀 [CURRICULUM] Масштаб {current_size} успешно освоен (Средняя награда: {avg_reward:.2f})!")
        print(f"📈 [CURRICULUM] Переводим систему на новый уровень: {new_size}x{new_size}...\n")
  
        ray.get(ga_manager.update_maze_size.remote(new_size))
        ray.get(ga_manager.generate_start_population.remote())

        algo.evaluation_config["env_config"]["maze_size"] = new_size
  
        algo.workers.foreach_worker(
            lambda worker: worker.foreach_env(
                lambda env: (
                    setattr(env, "maze_size", new_size),
                    setattr(env, "current_chromosome", None)
                )
            )
        )
        return new_size
    return current_size


if __name__ == "__main__":
    ray.init()

    CHECKPOINT_DIR = "models"
    GENERATIONS_LOG = "generations"
    EPOCHS = 256
    MAZE_SIZE = 8
    GENERATION_SIZE = 32
    STEPS_PER_CHROMOSOME = 128

    ga_manager = GAMazeManager.remote(GENERATION_SIZE, MAZE_SIZE)

    ModelCatalog.register_custom_model("maze_agent_cfc_model", RLTwoLayerCfcModel)

    config = (
        PPOConfig()
        .experimental(_enable_new_api_stack=False)
        .experimental(_disable_preprocessor_api=True)

        .environment(
            env=MazeEnvironment,
            env_config={
                "maze_size": MAZE_SIZE,
                "diamond_radius": 5,
                "maze_manager": ga_manager
            },
            normalize_actions=True
        )

        .callbacks(callbacks_class=lambda: CoEvolutionCallback(ga_manager))

        .framework("torch")

		.rollouts(
			num_rollout_workers=4,
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
                "custom_model": "maze_agent_cfc_model",
                "max_seq_len": 128,
                "use_lstm": False,
                "custom_model_config": {
                	"device": ("cuda:0" if torch.cuda.is_available() else "cpu")
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

    actual_start_ga = load_last_epoch(GENERATIONS_LOG, ga_manager)

    algo = config.build(logger_creator=lambda conf: NoopLogger(config=conf, logdir="/tmp", trial=None))

    start_generation_weights = load_pure_weights_to_algorithm(algo, CHECKPOINT_DIR)
    # load_model(algo, "last_model1.pth")

    actual_start = max(actual_start_ga, start_generation_weights)

    print(f"Старт обучения. Начинаем с поколения №{actual_start}")

    current_maze_size = MAZE_SIZE

    for generation in range(actual_start, EPOCHS):
        print(f"\n=== Начало поколения {generation} ===")

        ray.get(ga_manager.set_visits.remote())
        
        indx = 0
        while not ray.get(ga_manager.evolve_if_ready.remote(min_sample_per_chrome=STEPS_PER_CHROMOSOME)):
            indx += 1
            
            train_results = algo.train()
            
            avg_reward = train_results.get("episode_reward_mean", 0)
            
            custom_metrics = train_results.get("custom_metrics", {})
            avg_step_ratio = ray.get(ga_manager.get_avg_steps_ratio.remote())

            current_maze_size = check_and_update_curriculum(algo, ga_manager, current_maze_size, avg_step_ratio)

            print(f"Итерация обучения №{indx}. Средняя награда по батчу: {avg_reward:.2f}. Отношение идеала к результату: {avg_step_ratio:.2f}")

        save_pure_weights(algo, generation, CHECKPOINT_DIR)

        Path(GENERATIONS_LOG).mkdir(parents=True, exist_ok=True)
        population_data = ray.get(ga_manager.get_population.remote())
        fitness_scores = ray.get(ga_manager.get_fitness_scores.remote())

        data = [{"chromosome": p, "fitness_score": f, "maze_size": current_maze_size} for p, f in zip(ray.get(population_data), fitness_scores)]
        with open(Path(GENERATIONS_LOG).joinpath(f"generation_{generation}.json"), 'w') as file:
            json.dump(data, file, indent=4)

    algo.stop()
    ray.shutdown()

    print("Обучение завершено.")
