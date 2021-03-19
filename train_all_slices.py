import cvxpy as cp
import matplotlib.pyplot as matplt
from utils import *
from ddpg_alg_spinup import ddpg
import tensorflow as tf
from env_mra import ResourceEnv
import numpy as np
import time
import pickle
from parameters import *


if __name__ == "__main__":

    with open("saved_alpha.pickle", "wb") as fileop:
        pickle.dump(alpha, fileop)

    with open("saved_weight.pickle", "wb") as fileop:
        pickle.dump(weight, fileop)



    ########################################################################################################################
    ##########################################        Main Training           #############################################
    ########################################################################################################################
    start_time = time.time()
    utility = np.zeros(SliceNum)
    x = np.zeros([UENum, maxTime], dtype=np.float32)

    for i in range(SliceNum):

        ac_kwargs = dict(hidden_sizes=hidden_sizes, activation=tf.nn.relu, output_activation=tf.nn.sigmoid)

        logger_kwargs = dict(output_dir=str(RESNum)+'slice'+str(i), exp_name=str(RESNum)+'slice_exp'+str(i))

        env = ResourceEnv(alpha=alpha[i], weight=weight[i],
                          num_res=RESNum, num_user=UENum,
                          max_time=maxTime, min_reward=minReward,
                          rho=rho, test_env=False)

        utility[i], _ = ddpg(env=env, ac_kwargs=ac_kwargs,
                             steps_per_epoch=steps_per_epoch,
                             epochs=epochs, pi_lr=pi_lr, q_lr=q_lr,
                             start_steps=start_steps, batch_size=batch_size,
                             seed=seed, replay_size=replay_size, max_ep_len=maxTime,
                             logger_kwargs=logger_kwargs, fresh_learn_idx=True)

        print('slice' + str(i) + 'training completed.')


    end_time = time.time()
    print('Training Time is ' + str(end_time - start_time))

    #####################################          result ploting            ###############################################

    with open("saved_alpha.pickle", "rb") as fileop:
        load_alpha = pickle.load(fileop)

    with open("saved_weight.pickle", "rb") as fileop:
        load_weight = pickle.load(fileop)

    #print(weight)

    #matplt.subplot(2, 1, 1)
    #matplt.plot(sum_utility)
    #matplt.subplot(2, 1, 2)
    #matplt.plot(sum_x)
    matplt.show()

    print('done')


