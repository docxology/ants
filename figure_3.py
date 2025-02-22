import numpy as np
from pathlib import Path 
import matplotlib.pyplot as plt

import config as cf
from ants import main

Path("imgs").mkdir(parents=True, exist_ok=True)

NUM_AVERAGE = 5
NUM_STEPS = 100
ANT_CONFIGS = [(10, "compare_strict_10"), (30, "compare_strict_30"), (50, "compare_strict_50"), (70, "compare_strict_70")]

C = np.array([[0, 0, 0, 0, 0, 1, 2, 3, 4, 5]])

def run(init_ants, name):
    num_round_trips, paths, coeff = main(
        num_steps=NUM_STEPS,
        init_ants=init_ants,
        max_ants=init_ants,
        C=C,
        save=True,
        switch=True,
        name=name,
    )
    print(f"num_round_trips_{init_ants} {num_round_trips} / coeff {coeff}")
    with open(f"imgs/{name}.txt", "a+") as f:
        f.write(f"num_round_trips_{init_ants} {num_round_trips} / coeff {coeff}\n")
    # Visualization
    plt.figure(figsize=(10, 6))
    for i, path in enumerate(np.random.choice(paths, size=min(len(paths), 5), replace=False)):
        plt.plot(*zip(*path), marker='o', linestyle='-', label=f"Path {i+1}")
    plt.title(f"Paths Visualization for {init_ants} Ants")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.legend()
    plt.savefig(f"imgs/{name}_paths.png")
    plt.close()

if __name__ == "__main__":
    for _ in range(NUM_AVERAGE):
        for ants, name in ANT_CONFIGS:
            run(ants, name)

