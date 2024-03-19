import numpy as np
from pathlib import Path 

import config as cf
from ants import main

# Ensure the directory for images exists
Path("imgs").mkdir(parents=True, exist_ok=True)

# Constants
NUM_AVERAGE = 5
NUM_STEPS = 100
ANT_CONFIGS = [(10, "compare_10"), (30, "compare_30"), (50, "compare_50"), (70, "compare_70")]

# Standard prior
PRIOR_TICK = 1
C = np.zeros((cf.NUM_OBSERVATIONS, 1))
C[:, 0] = np.arange(0, cf.NUM_OBSERVATIONS * PRIOR_TICK, PRIOR_TICK)

def run_simulation(init_ants, name):
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

if __name__ == "__main__":
    for _ in range(NUM_AVERAGE):
        for ants, name in ANT_CONFIGS:
            run_simulation(ants, name)

