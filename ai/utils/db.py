from pathlib import Path
import json
import bsc_core

path = Path("../generations")
target_path = Path("DB")

for k, p in enumerate(path.iterdir()):
    with open(p, 'r') as file:
        data = json.load(file)

    cur_dir = target_path.joinpath(f"generation_{k}")
    cur_dir.mkdir(parents=True, exist_ok=True)

    for l, e in enumerate(data):
        maze = bsc_core.generate_with_genome(e["chromosome"], 128)

        with open(cur_dir.joinpath(f"chromosome_{l + 1}.json"), 'w') as file:
            json.dump(maze, file)
