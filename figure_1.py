import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import config as cf
from ants import main, plot_path

# Define constants
NAME = "main"
NUM_STEPS = 100
INIT_ANTS = 10
MAX_ANTS = 10
IMG_DIR = Path("imgs/")

# Ensure the image directory exists
IMG_DIR.mkdir(parents=True, exist_ok=True)

# Initialize the prior matrix
PRIOR_TICK = 1
C = np.linspace(0, PRIOR_TICK * (cf.NUM_OBSERVATIONS - 1), cf.NUM_OBSERVATIONS).reshape(-1, 1)

if __name__ == "__main__":
    # Run the main simulation and get results
    num_round_trips, paths, coeff = main(
        num_steps=NUM_STEPS,
        init_ants=INIT_ANTS,
        max_ants=MAX_ANTS,
        C=C,
        save=True,
        switch=True,
        name=NAME,
        ant_only_gif=False,
    )
    
    # Log the results
    result_str = f"num_round_trips {num_round_trips} / coeff {coeff / MAX_ANTS}"
    print(result_str)
    with open(IMG_DIR / f"{NAME}.txt", "w") as f:
        f.write(result_str)
    
    # Plot and save paths
    selected_paths = np.random.choice(paths, size=min(len(paths), 5), replace=False)
    for i, path in enumerate(selected_paths):
        fig, ax = plt.subplots()
        ax.plot(*zip(*path), marker='o', color='r', ls='-')
        ax.set_title(f"Path {i+1}")
        plt.savefig(IMG_DIR / f"path_{i}.png")
        plt.close(fig)

