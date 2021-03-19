import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import matplotlib.pyplot as matplt

class ResourceEnv(gym.Env):

    def __init__(self, total_resource=1.0, num_user=5, min_reward=100, max_time=100, rho=0.01, aug_penalty=1.0, random_seed=55674):
        self.ResNum = total_resource  # total number of resource
        self.UENum = num_user  # number of slices
        self.maxTime = max_time
        self.min_Reward = min_reward * np.ones(self.UENum)
        self.random_seed = random_seed
        self.aug_penalty = aug_penalty
        self.rho = rho
        self.beta = 0

        np.random.seed(self.random_seed)
        self.weight = np.random.uniform(0.01, 0.99, self.UENum)

        self.action_min = np.zeros(self.UENum)
        self.action_max = np.ones(self.UENum)
        self.state_min = np.zeros(self.UENum)
        self.state_max = np.ones(self.UENum)

        self.action_space = spaces.Box(self.action_min, self.action_max, dtype=np.float32)
        self.observation_space = spaces.Box(-self.state_max, self.state_max, dtype=np.float32)

        self.alpha = self.generate_alpha_function(self.UENum)
        #matplt.show()
        self.seed()

        # these variables need reset for env
        self.iter = 1
        self.accu_reward = np.zeros(self.UENum)

        self.reset()

    def seed(self, seed=None):
        _, seed = seeding.np_random(seed)
        return [seed]

    def step(self, in_action):

        action = np.clip(in_action, 0, 1)
        action *= self.action_max  # scale to max

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        real_reward = self.calculate_reward(action)

        penalty = self.rho * np.abs(self.aug_penalty - np.sum(action))  # should be square but too small when gap is 0.1, not good for convergence

        weight_reward = np.multiply(real_reward, self.weight)

        self.accu_reward = np.add(self.accu_reward, real_reward)

        remaining_reward = np.clip(np.subtract(self.min_Reward, self.accu_reward), 0, None) / self.min_Reward

        final_state = remaining_reward

        final_reward = np.sum(weight_reward - self.iter * remaining_reward) - penalty  # increase the weight of penalty when the episode is almost done

        self.iter += 1
        done = False
        if self.iter >= self.maxTime:
            done = True
            final_state = self.reset()
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
            #reward[i] = - action[i] * action[i] + 2 * self.alpha[i] * action[i] + 2
            # in case action is really 100
            #reward[i] = self.functions[i][min(int(action[i]), self.ResNum-1)]

        return reward

    def generate_alpha_function(self, num_user, min_val=0.1, max_val=1,):
        np.random.seed(self.random_seed)
        alpha = np.random.uniform(min_val, max_val, num_user)
        return alpha

    def generate_alpha_fairness_function(self, resolution, num_user, min_val=0, max_val=1,):
        FX = []
        np.random.seed(self.random_seed)
        for i in range(num_user):
            x = np.linspace(0, 1, resolution)
            alpha = np.random.uniform(0, 1)
            fx = (x**alpha)/alpha

            fx = min_val + fx - min(fx)  # normalize to min_val
            fx = max_val * fx/max(fx)   # normalize to max_val

            matplt.plot(fx)
            FX.append(fx)
        return FX

    def reset(self):
        self.iter = 1
        self.accu_reward = np.zeros(self.UENum)
        initial_state = np.ones(self.UENum)
        return initial_state

    def generate_random_para(self):
        np.random.seed(self.random_seed)
        alpha = np.random.uniform(0, 1, self.UENum)
        return alpha

    def render(self):
        pass

    def close(self):
        pass
