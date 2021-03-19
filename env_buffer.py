import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as matplt


class ResourceEnv(gym.Env):

    def __init__(self, alpha, weight, total_resource=100, num_user=5, buffer_size=200, buffer = 100, rho=1.0, aug_penalty=1.0,):
        self.Rmax = total_resource  # total number of resource
        self.UENum = num_user  # number of slices
        self.aug_penalty = aug_penalty
        self.rho = rho
        self.alpha = alpha
        self.weight = weight
        self.buffer_size = buffer_size

        self.action_min = np.zeros(self.UENum)
        self.action_max = np.ones(self.UENum)
        self.state_min = np.zeros(self.UENum)
        self.state_max = np.ones(self.UENum)

        self.action_space = spaces.Box(self.action_min, self.action_max, dtype=np.float32)
        self.observation_space = spaces.Box(self.state_min, self.state_max, dtype=np.float32)

        # these variables need reset for env
        self.buffer = buffer_size * np.ones(self.UENum)

        self.reset()


    def step(self, in_action):

        action = np.clip(in_action, self.action_min, self.action_max)

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        action *= self.Rmax

        real_reward = self.calculate_reward(action)

        self.buffer = np.clip(np.subtract(self.buffer, real_reward), 0, None)

        weight_reward = np.multiply(real_reward, self.weight)

        augmented_reward = np.multiply(weight_reward, self.buffer/self.buffer_size)

        final_state = self.buffer

        penalty = 0.5 * self.rho * np.abs(np.sum(action) - self.aug_penalty)  # should be square but too small when gap is 0.1, not good for convergence

        final_reward = np.sum(weight_reward) - penalty  # increase the weight of penalty when the episode is almost done

        done = False

        #if np.sum(final_state) == 0:
            #done = True
            #final_state = self.reset()

        return final_state, final_reward, done, np.sum(weight_reward)

    def calculate_reward(self, action):

        reward = np.zeros(self.UENum)

        for i in range(self.UENum):
            reward[i] = (action[i] ** self.alpha[i]) / self.alpha[i]

        return reward


    def reset(self):

        self.buffer = self.buffer_size * np.ones(self.UENum)

        return self.buffer

    def render(self):
        pass

    def close(self):
        pass
