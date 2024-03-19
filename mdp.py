import numpy as np
import config as cf

class MDP(object):
    def __init__(self, A, B, C):
        self.A = A
        self.B = B
        self.C = C
        self.p0 = np.exp(-16)

        self.num_states = self.A.shape[1]
        self.num_obs = self.A.shape[0]
        self.num_actions = self.B.shape[0]

        self.A = self.A + self.p0
        self.A = self.normdist(self.A)
        self.lnA = np.log(self.A)

        self.B = self.B + self.p0
        for a in range(self.num_actions):
            self.B[a] = self.normdist(self.B[a])

        self.C = self.C + self.p0
        self.C = self.normdist(self.C)

        self.sQ = np.zeros([self.num_states, 1])
        self.uQ = np.zeros([self.num_actions, 1])
        self.prev_action = None
        self.t = 0

    def set_A(self, A):
        self.A = A + self.p0
        self.A = self.normdist(self.A)
        self.lnA = np.log(self.A)

    def reset(self, obs):
        self.t = 0
        self.curr_obs = obs
        likelihood = self.lnA[obs, :]
        likelihood = likelihood[:, np.newaxis]
        self.sQ = self.softmax(likelihood)
        self.prev_action = self.random_action()

    def step(self, obs):
        """ state inference """
        likelihood = self.lnA[obs, :]
        likelihood = likelihood[:, np.newaxis]
        prior = np.dot(self.B[self.prev_action], self.sQ)
        prior = np.log(prior)
        self.sQ = self.softmax(prior)

        """ action inference """
        SCALE = 10
        neg_efe = np.zeros([self.num_actions, 1])
        for a in range(self.num_actions):
            fs = np.dot(self.B[a], self.sQ)
            fo = np.dot(self.A, fs)
            fo = self.normdist(fo + self.p0)
            utility = np.sum(fo * np.log(fo / self.C), axis=0)
            utility = utility[0]
            neg_efe[a] -= utility / SCALE

        # priors
        neg_efe[4] -= 20.0
        neg_efe[cf.OPPOSITE_ACTIONS[self.prev_action]] -= 20.0 

        # action selection
        self.uQ = self.softmax(neg_efe)
        action = np.argmax(np.random.multinomial(1, self.uQ.squeeze()))
        self.prev_action = action
        return action

    def random_action(self):
        return int(np.random.choice(range(self.num_actions)))

    @staticmethod
    def softmax(x):
        x = x - x.max()
        x = np.exp(x)
        x = x / np.sum(x)
        return x

    @staticmethod
    def normdist(x):
        return np.dot(x, np.diag(1 / np.sum(x, 0)))
