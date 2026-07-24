import matplotlib
matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import bsc_core
import json
from pathlib import Path

from matplotlib.colors import ListedColormap

MAZE_SIZE = 8
GENERATIONS_DIR = "../generations"
GENDEMO_DIR = "./demo_generations/"

cur_generations, cur_genome = 0, 0

generations_path = Path(GENERATIONS_DIR)
gendemo_path = Path(GENDEMO_DIR)


def get_generation_number(path_obj):
    try:
        return int(''.join(filter(str.isdigit, path_obj.name)))
    except ValueError:
        return -1


valid_generations = [p for p in generations_path.iterdir() if "generation_" in p.name]
for generation in sorted(valid_generations, key=get_generation_number):
    try:
        g = int(''.join(filter(str.isdigit, generation.name)))
    except ValueError:
        continue
    
    if g < cur_generations:
        continue

    with open(str(generation), 'r') as file:
        population_data = json.load(file)
        
    chromosomes = [(e["chromosome"], e["fitness_score"]) for e in population_data]
        
    print(f"=== Генерация изображений поколения №{g} ===")
    gendemo_path.joinpath(f"generation{g}").mkdir(parents=True, exist_ok=True)
        
    for k, (chromosome, fitness_score) in enumerate(chromosomes):
        if g == cur_generations and k + 1 < cur_genome:
            continue
    
        print(f"Отрисовка генома №{k + 1}...")
        maze_dict = bsc_core.generate_with_genome(chromosome, MAZE_SIZE)

        matrix = maze_dict.get("mat")
        start_pos = maze_dict.get("start")
        finish_pos = maze_dict.get("finish")
        maze_cmap = ListedColormap(['#1a1a1a', '#ffffff', '#2ecc71'])

        fig, ax = plt.subplots(figsize=(10, 10), dpi=150)
        
        ax.imshow(matrix, cmap=maze_cmap, origin="upper")

        start_x = start_pos[1]
        start_y = start_pos[0]
        
        finish_x = finish_pos[1]
        finish_y = finish_pos[0]
        
        first_dot = chromosome.find('.')
        second_dot = chromosome.find('.', first_dot + 1)
        
        gen_x = int(chromosome[first_dot + 1: second_dot])
        gen_y = int(chromosome[:first_dot])

        ax.scatter(start_x, start_y, color="green", s=150, marker='s', label="Старт агента", alpha=0.7)
        ax.scatter(finish_x, finish_y, color="red", s=150, marker='x', label="Финиш агента")

        ax.scatter(gen_x, gen_y, color="blue", s=50, marker="o", alpha=0.6, label=f"Точка зарождения ГЛ ({gen_x}, {gen_y})")

        ax.set_title(f"Коэволюционный лабиринт №{k + 1} поколения {g}", fontsize=14, pad=15)
        fig.text(0.68, 0.14, f"Fitness Score: {fitness_score:.2f}", 
                 fontsize=12, fontweight='bold', color='black',
                 bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', boxstyle='round,pad=0.5'))

        ax.legend(loc="upper right", fontsize=10)
        ax.axis("off")

        output_path = gendemo_path.joinpath(f"generation{g}").joinpath(f"genome{k + 1}.png")
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

