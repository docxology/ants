import numpy as np
import config as cf

import matplotlib
import matplotlib.pyplot as plt
import imageio

matplotlib.use("Agg")

class Env(object):
    def __init__(self):
        self.visit_matrix = np.zeros((cf.GRID[0], cf.GRID[1]))
        self.obs_matrix = np.zeros((cf.NUM_OBSERVATIONS, cf.GRID[0], cf.GRID[1]))
        self.obs_matrix[0, :, :] = 1.0

    def get_A(self, ant):
        A = np.zeros((cf.NUM_OBSERVATIONS, cf.NUM_STATES))
        for s in range(cf.NUM_STATES):
            delta = cf.ACTION_MAP[s]
            A[:, s] = self.obs_matrix[:, ant.x_pos + delta[0], ant.y_pos + delta[1]]
        return A

    def get_obs(self, ant):
        obs_vec = self.obs_matrix[:, ant.x_pos, ant.y_pos]
        return np.argmax(obs_vec)

    def check_food(self, x_pos, y_pos):
        is_food = False
        if (x_pos > (cf.FOOD_LOCATION[0] - cf.FOOD_SIZE[0])) and (
            x_pos < (cf.FOOD_LOCATION[0] + cf.FOOD_SIZE[0])
        ):
            if (y_pos > (cf.FOOD_LOCATION[1] - cf.FOOD_SIZE[1])) and (
                y_pos < (cf.FOOD_LOCATION[1] + cf.FOOD_SIZE[1])
            ):
                is_food = True
        return is_food

    def check_walls(self, orig_x, orig_y, x_pos, y_pos):
        valid = True
        if orig_y > cf.WALL_TOP:
            if orig_x >= cf.WALL_LEFT and x_pos <= cf.WALL_LEFT:
                valid = False
            if orig_x <= cf.WALL_RIGHT and x_pos >= cf.WALL_RIGHT:
                valid = False
        if orig_y <= cf.WALL_TOP:
            if y_pos > cf.WALL_TOP and ((x_pos < cf.WALL_LEFT) or (x_pos > cf.WALL_RIGHT)):
                valid = False
        return valid

    def step_forward(self, ant, action):
        delta = cf.ACTION_MAP[action]
        x_pos = np.clip(ant.x_pos + delta[0], 1, cf.GRID[0] - 2)
        y_pos = np.clip(ant.y_pos + delta[1], 1, cf.GRID[1] - 2)

        if self.check_food(x_pos, y_pos) and np.random.rand() < cf.NEST_FACTOR:
            ant.is_returning = True
            ant.backward_step = 0

        if self.check_walls(ant.x_pos, ant.y_pos, x_pos, y_pos):
            ant.update_forward(x_pos, y_pos)

    def step_backward(self, ant):
        path_len = len(ant.traj)
        next_step = path_len - (ant.backward_step + 1)
        pos = ant.traj[next_step]
        ant.update_backward(pos[0], pos[1])

        self.visit_matrix[pos[0], pos[1]] += 1
        curr_obs = np.argmax(self.obs_matrix[:, pos[0], pos[1]])
        curr_obs = min(curr_obs + 1, cf.NUM_OBSERVATIONS - 1)

        self.obs_matrix[:, pos[0], pos[1]] = 0.0
        self.obs_matrix[curr_obs, pos[0], pos[1]] = 1.0

        ant.backward_step += 1
        if ant.backward_step >= path_len - 1:
            ant.is_returning = False
            traj = ant.traj
            ant.traj = []
            return True, traj
        else:
            return False, None

    def decay(self):
        for x in range(cf.GRID[0]):
            for y in range(cf.GRID[1]):
                curr_obs = np.argmax(self.obs_matrix[:, x, y])
                if (curr_obs > 0) and (np.random.rand() < cf.DECAY_FACTOR):
                    curr_obs = curr_obs - 1
                    self.obs_matrix[:, x, y] = 0.0
                    self.obs_matrix[curr_obs, x, y] = 1.0

    def plot(self, ants, savefig=False, name="", ant_only_gif=False):
        x_pos_forward, y_pos_forward = [], []
        x_pos_backward, y_pos_backward = [], []
        for ant in ants:
            if ant.is_returning:
                x_pos_backward.append(ant.x_pos)
                y_pos_backward.append(ant.y_pos)
            else:
                x_pos_forward.append(ant.x_pos)
                y_pos_forward.append(ant.y_pos)

        img = np.ones((cf.GRID[0], cf.GRID[1]))
        fig, ax = plt.subplots()
        ax.imshow(img.T, cmap="gray")
        plot_matrix = np.zeros((cf.GRID[0], cf.GRID[1]))

        for x in range(cf.GRID[0]):
            for y in range(cf.GRID[1]):
                curr_obs = np.argmax(self.obs_matrix[:, x, y])
                plot_matrix[x, y] = curr_obs

        if not ant_only_gif:
            ax.imshow(plot_matrix.T, alpha=0.7, vmin=0)

        if not savefig:
            ax.scatter(x_pos_forward, y_pos_forward, color="red", s=5)
            ax.scatter(x_pos_backward, y_pos_backward, color="blue", s=5)

        if not savefig:
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close("all")
            return img
        else:
            plt.savefig(name)
        plt.close("all")