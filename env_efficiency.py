import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class ResourceEnv(gym.Env):

    def __init__(self, max_resource=1, num_slice=5, min_reward=100, max_time=100, big_reward=1e4, random_seed=123):
        self.ResourceNum = max_resource  # total number of resource
        self.NUM = num_slice  # number of slices
        self.maxTime = max_time
        self.big_reward = big_reward
        self.min_Reward = min_reward * np.ones(self.NUM)
        self.random_seed = random_seed
        self.beta = 0

        self.action_min = -np.ones(self.NUM)
        self.action_max = self.ResourceNum * np.ones(self.NUM)
        self.state_min = np.zeros(self.NUM)
        self.state_max = np.inf * np.ones(self.NUM)

        self.action_space = spaces.Box(self.action_min, self.action_max, dtype=np.float32)
        self.observation_space = spaces.Box(-self.state_max, self.state_max, dtype=np.float32)

        # these variables need reset for env
        self.iter = 1
        self.alpha = self.generate_random_para()
        self.accu_reward = np.zeros(self.NUM)
        self.remaining_reward = self.min_Reward
        self.state = self.remaining_reward

        self.seed()
        self.reset()

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return [seed]

    def step(self, in_action):
        action = in_action
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        reward = self.reward_function(action)

        self.accu_reward = np.add(self.accu_reward, reward)

        self.remaining_reward = np.clip(np.subtract(self.min_Reward, self.accu_reward), 0, None)

        self.state = self.remaining_reward

        final_reward = -np.sum(self.remaining_reward)

        self.iter += 1
        done = False
        if (self.iter >= self.maxTime) | (final_reward == 0):
            done = True
            final_reward = self.big_reward

        #info = dict(real_reward=np.sum(reward), penalty=penalty)

        return self.state, final_reward, done, []

    def determine_constraint_penalty(self):
        constraint_status = np.clip(np.divide(self.accu_reward, self.min_Reward), 0, 1)
        penalty = np.zeros(self.NUM)
        for i in range(self.NUM):
            penalty[i] = 1 - np.sqrt(1 - (1 - constraint_status[i])**2)
        return constraint_status, penalty

    def reward_function(self, action):
        reward = np.zeros(self.NUM)
        for i in range(self.NUM):
            #reward[i] = (action[i] ** (1 - self.alpha[i])) / (1 - self.alpha[i])
            reward[i] = action[i] * (action[i] - 2 * self.alpha[i]) + 1
            #reward[i] = (self.alpha[i] - action[i]) * action[i]
        return reward

    def reset(self):
        self.iter = 1
        self.accu_reward = np.zeros(self.NUM)
        self.state = np.zeros(self.NUM)
        return self.state

    def generate_random_para(self):
        np.random.seed(self.random_seed)
        alpha = np.random.uniform(0, 1, self.NUM)
        return alpha

    def render(self):
        pass

    def close(self):
        pass
