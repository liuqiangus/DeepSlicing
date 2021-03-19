import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as matplt
from parameters import use_other_utility_function

class ResourceEnv(gym.Env):

    def __init__(self, alpha, weight, total_resource=100, num_res=3, num_user=5, min_reward=100, max_time=20, rho=1.0, aug_penalty=[], test_env=False):
        self.Rmax = total_resource  # total number of resource
        self.UENum = num_user  # number of slices
        self.num_res = num_res
        self.maxTime = max_time
        self.min_Reward = min_reward * np.ones(self.UENum)
        self.aug_penalty = self.Rmax * np.ones(self.num_res)
        self.aug_penalty = aug_penalty
        self.rho = rho
        self.alpha = alpha
        self.weight = weight
        self.test_env = test_env

        self.action_min = np.zeros(self.UENum*self.num_res)
        self.action_max = np.ones(self.UENum*self.num_res)
        self.state_min = np.zeros(self.UENum+self.num_res)
        self.state_max = np.ones(self.UENum+self.num_res)

        self.action_space = spaces.Box(self.action_min, self.action_max, dtype=np.float32)
        self.observation_space = spaces.Box(self.state_min, self.state_max, dtype=np.float32)

        self.action_dim = self.action_space.shape[0]
        self.observe_dim = self.observation_space.shape[0]

        # these variables need reset for env
        self.iter = 0
        self.accu_reward = np.zeros(self.UENum)
        self.remain_reward = self.min_Reward

        self.reset()


    def step(self, in_action):

        action = np.clip(in_action, self.action_min, self.action_max)

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        action = self.Rmax * np.reshape(action, [self.num_res, self.UENum])  # reshape into number of resource * number of users

        penalty = 0.5 * self.rho * np.sum(np.abs(np.sum(action, axis=1) - self.aug_penalty))  # should be square but too small when gap is 0.1, not good for convergence

        real_reward = self.calculate_reward(action)

        weight_reward = np.multiply(real_reward, self.weight)

        self.accu_reward = np.add(self.accu_reward, real_reward)

        #constraint = np.clip((self.remain_reward / (self.maxTime - self.iter)) - real_reward, 0, None)  # avg_reward_until_now

        #constraint = (self.remain_reward / (self.maxTime - self.iter)) - real_reward  # avg_reward_until_now

        constraint = [self.tanh_func(real_reward[i], self.min_Reward[i] / self.maxTime) for i in range(self.UENum)]  # avg_reward_until_now

        self.remain_reward = np.clip(np.subtract(self.min_Reward, self.accu_reward), 0, None)

        final_state = np.concatenate([self.remain_reward, self.aug_penalty])

        # use maxTime as the weight of constraint, since its calculation is some kind of divided by maxTime
        final_reward = np.sum(weight_reward) + self.maxTime * np.sum(constraint) - penalty  # increase the weight of penalty when the episode is almost done

        self.iter += 1

        done = False

        if self.iter >= self.maxTime:
            done = True
            self.reset()

        return final_state, final_reward, done, np.sum(weight_reward)

    def calculate_reward(self, action):

        reward = np.zeros([self.num_res, self.UENum], dtype=np.float32)

        for i in range(self.num_res):
            for j in range(self.UENum):
                reward[i][j] = (action[i][j] ** self.alpha[i][j]) / self.alpha[i][j]
                if use_other_utility_function:
                    reward[i][j] = self.Rmax/(self.Rmax * np.exp(- self.alpha[i][j] * action[i][j]) + 1)

        return np.mean(reward, axis=0)   # np.min(reward, axis=0)

    def tanh_func(self, x, a):

        y = np.clip(1/(np.exp(-2*(x-a))) - 1, -1, 0)

        return y

    def reset(self):

        self.iter = 0

        self.accu_reward = np.zeros(self.UENum)

        self.remain_reward = self.min_Reward

        if not self.test_env:  # if not test the env, we random the penalty for training
            self.aug_penalty = np.random.uniform(0, self.Rmax, self.num_res)

        initial_state = np.concatenate([self.remain_reward, self.aug_penalty])

        return initial_state

    def render(self):
        pass

    def close(self):
        pass
