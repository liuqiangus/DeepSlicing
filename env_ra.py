import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as matplt


class ResourceEnv(gym.Env):

    def __init__(self, alpha, weight, total_resource=100, num_user=5, min_reward=200, max_time=100, rho=1.0, aug_penalty=100, test_env=False):
        self.Rmax = total_resource  # total number of resource
        self.UENum = num_user  # number of slices
        self.maxTime = max_time
        self.min_Reward = min_reward * np.ones(self.UENum)
        self.aug_penalty = aug_penalty
        self.rho = rho
        self.alpha = alpha
        self.weight = weight
        self.test_env = test_env

        self.action_min = np.zeros(self.UENum)
        self.action_max = np.ones(self.UENum)
        self.state_min = np.zeros(self.UENum+1)
        self.state_max = self.Rmax * np.ones(self.UENum+1)

        self.action_space = spaces.Box(self.action_min, self.action_max, dtype=np.float32)
        self.observation_space = spaces.Box(self.state_min, self.state_max, dtype=np.float32)

        # these variables need reset for env
        self.iter = 0
        self.accu_reward = np.zeros(self.UENum)
        self.remain_reward = self.min_Reward

        self.reset()


    def step(self, in_action):

        action = np.clip(in_action, self.action_min, self.action_max)

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        action *= self.Rmax

        real_reward = self.calculate_reward(action)

        penalty = 0.5 * self.rho * np.abs(np.sum(action) - self.aug_penalty)  # should be square but too small when gap is 0.1, not good for convergence

        weight_reward = np.multiply(real_reward, self.weight)

        self.accu_reward = np.add(self.accu_reward, real_reward)

        if self.maxTime == self.iter:
            print('1')

        constraint = np.clip((self.remain_reward / (self.maxTime - self.iter)) - real_reward, 0, None)  # avg_reward_until_now

        self.remain_reward = np.clip(np.subtract(self.min_Reward, self.accu_reward), 0, None)

        final_state = np.concatenate([self.remain_reward, [self.aug_penalty]])

        #final_reward = np.sum(weight_reward - self.iter * constraint) - penalty  # increase the weight of penalty when the episode is almost done
        final_reward = np.sum(weight_reward) - np.sum(constraint) - penalty  # increase the weight of penalty when the episode is almost done

        self.iter += 1

        done = False

        if self.iter >= self.maxTime:
            done = True
            self.reset()
        #elif final_reward == 0:
        #    done = True
        #    final_reward = self.big_reward
        #    final_state = self.reset()

        #info = dict(real_reward=np.sum(reward), penalty=penalty)

        return final_state, final_reward, done, np.sum(weight_reward)

    def calculate_reward(self, action):

        reward = np.zeros(self.UENum)

        for i in range(self.UENum):
            reward[i] = (action[i] ** self.alpha[i]) / self.alpha[i]

        return reward


    def reset(self):

        self.iter = 0

        self.accu_reward = np.zeros(self.UENum)

        self.remain_reward = self.min_Reward

        if not self.test_env:  # if not test the env, we random the penalty for training
            self.aug_penalty = np.random.uniform(0, self.Rmax)

        initial_state = np.concatenate([self.remain_reward, [self.aug_penalty]])

        return initial_state

    def render(self):
        pass

    def close(self):
        pass
