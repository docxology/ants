import numpy as np
import config as cf

from mdp import MDP 

class Ant(object):
    def __init__(self, mdp, init_x, init_y):
        self.mdp = mdp
        self.x_pos = init_x
        self.y_pos = init_y
        self.traj = [(init_x, init_y)]
        self.distance = []
        self.backward_step = 0
        self.is_returning = False

    @staticmethod
    def dis(x1, y1, x2, y2):
        return np.sqrt(((x1 - x2) ** 2) + ((y1 - y2) ** 2))

    def update_forward(self, x_pos, y_pos):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.traj.append((x_pos, y_pos))
        self.distance.append(Ant.dis(x_pos, y_pos, cf.INIT_X, cf.INIT_Y))

    def update_backward(self, x_pos, y_pos):
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.distance.append(Ant.dis(x_pos, y_pos, cf.INIT_X, cf.INIT_Y))

    @classmethod
    def create(cls, init_x, init_y, C):
        A = np.zeros((cf.NUM_OBSERVATIONS, cf.NUM_STATES))
        B = np.zeros((cf.NUM_ACTIONS, cf.NUM_STATES, cf.NUM_STATES))
        for a in range(cf.NUM_ACTIONS):
            B[a, a, :] = 1.0
        mdp = MDP(A, B, C)
        return cls(mdp, init_x, init_y)
