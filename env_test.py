from env_mra import ResourceEnv
import numpy as np
import matplotlib.pyplot as matplt
from utils import generate_alpha_fairness_function
from ddpg_alg_spinup import ReplayBuffer

UENum = 5
RESNum = 3
seed = 123456

np.random.seed(seed)
alpha = np.random.uniform(0.1, 0.9, [RESNum, UENum])
np.random.seed(seed)
weight = np.random.uniform(0.1, 0.9, UENum)


env = ResourceEnv(alpha=alpha, weight=weight, rho=0.1, test_env=True,)
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]

#replaybuffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=int(1e6))


#################################### random generate solution ############################################
total_reward = []
total_info = []
total_action = []
state = env.reset()
for ite in range(int(5e4)):
    print(ite)
    ep_rew, ep_real = 0, 0

    for i in range(100):

        action = env.action_space.sample()

        #action = action / np.sum(action)

        state_, reward, done, info = env.step(action)

        ep_rew += reward
        ep_real += info

        #replaybuffer.store(state, action, reward, state_, done)

        state = state_
        #print('[', i, ']', state, action, reward, info)
        if done:
            break
    total_reward.append(ep_rew)
    total_info.append(ep_real)
print('done iteration %d' % i, 'total_reward %.2f' % np.sum(total_reward), 'total_info %.2f' % np.sum(total_info))


#replaybuffer.store_buffer_into_file()



print('done')

matplt.plot(total_reward)
matplt.show()
