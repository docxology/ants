import config as cf
import numpy as np

from ant import Ant
from env import Env
from mdp import MDP
from visualize import plot_path, save_gif
from pathlib import Path

def main(num_steps, init_ants, max_ants, C, save=True, switch=False, name="", ant_only_gif=False):
    env = Env()
    ants = []
    paths = []
    for _ in range(init_ants):
        ant = Ant.create(cf.INIT_X, cf.INIT_Y, C)
        obs = env.get_obs(ant)
        A = env.get_A(ant)
        ant.mdp.set_A(A)
        ant.mdp.reset(obs)
        ants.append(ant)

    imgs = []
    completed_trips = 0
    distance = 0
    ant_locations = []
    round_trips_over_time = []
    
    # Create a sub-folder for the current run
    img_subfolder_path = Path(f"imgs/{name}")
    img_subfolder_path.mkdir(parents=True, exist_ok=True)

    for t in range(num_steps):
        t_dis = 0

        for ant in ants:
            for ant_2 in ants:
                t_dis += Ant.dis(ant.x_pos, ant.y_pos, ant_2.x_pos, ant_2.y_pos)
        distance += t_dis / len(ants)

        if t % (num_steps // 100) == 0:
            print(f"{t}/{num_steps}")

        if t % cf.ADD_ANT_EVERY == 0 and len(ants) < max_ants:
            ant = Ant.create(cf.INIT_X, cf.INIT_Y, C)
            obs = env.get_obs(ant)
            A = env.get_A(ant)
            ant.mdp.set_A(A)
            ant.mdp.reset(obs)
            ants.append(ant)

        if switch and t % (num_steps // 2) == 0:
            # switch
            cf.FOOD_LOCATION = (cf.GRID[0] - cf.FOOD_LOCATION[0], cf.FOOD_LOCATION[1])

        for ant in ants:
            if not ant.is_returning:
                obs = env.get_obs(ant)
                A = env.get_A(ant)
                ant.mdp.set_A(A)
                action = ant.mdp.step(obs)
                env.step_forward(ant, action)
            else:
                is_complete, traj = env.step_backward(ant)
                completed_trips += int(is_complete)

                if is_complete:
                    paths.append(traj)
        env.decay()

        if save:
            if t in np.arange(0, num_steps, num_steps // 20):
                env.plot(ants, savefig=True, name=f"{img_subfolder_path}/{t}.png")
            else:
                img = env.plot(ants, ant_only_gif=ant_only_gif)
                imgs.append(img)

        round_trips_over_time.append(completed_trips / max_ants)
        ant_locations.append([[ant.x_pos, ant.y_pos] for ant in ants])

    if save:
        save_gif(imgs, f"{img_subfolder_path}/{name}.gif")

    ant_locations = np.array(ant_locations)
    round_trips_over_time = np.array(round_trips_over_time)
    np.save(f"{img_subfolder_path}/{name}_locations", ant_locations)
    np.save(f"{img_subfolder_path}/{name}_round_trips", round_trips_over_time)

    return completed_trips, np.array(paths), distance
