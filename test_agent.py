import numpy as np
import time
import pickle
from functions import *
import sys


if __name__ == "__main__":

    agent_id = 1

    ########################################################################################################################
    ##########################################        Main Simulation          #############################################
    ########################################################################################################################

    start_time = time.time()
    sum_gap= []

    aug_utility_opt, aug_utility_ddpg = [], []
    real_utility_opt, real_utility_ddpg, real_utility_static = [], [], []
    x_opt = np.zeros([RESNum, UENum], dtype=np.float32)
    x_ddpg = np.zeros([RESNum, UENum], dtype=np.float32)
    x_static = np.zeros([RESNum, UENum], dtype=np.float32)

    for ite in range(50):

        ###################### random ADMM penalty #########################################
        z_minus_u = np.random.uniform(Rmin, Rmax, RESNum)

        tmp_utility, tmp_real_utility = np.zeros(RESNum), np.zeros(RESNum)


        ############################  static agent  ##########################################
        for j in range(RESNum):

            tmp_utility[j], x_static[j], tmp_real_utility[j] = \
                    simple_static_alogrithm(z_minus_u=z_minus_u[j],
                                            alpha=alpha[agent_id, j],
                                            weight=weight[agent_id], UENum=UENum,
                                            minReward=minReward/maxTime)

        real_utility_static.append(tmp_real_utility * maxTime)

        ############################  optimization  ########################################

        for j in range(RESNum):

            # since all the conditions are the same for all time slots, we assign the same results to all time slots
            tmp_utility[j], x_opt[j], tmp_real_utility[j] = \
                    simple_convex_alogrithm(z_minus_u=z_minus_u[j],
                                            alpha=alpha[agent_id, j],
                                            weight=weight[agent_id], UENum=UENum,
                                            minReward=minReward/maxTime)

        aug_utility_opt.append(np.mean(tmp_utility) * maxTime)  # utility of slice -- mean for all resources

        real_utility_opt.append(np.mean(tmp_real_utility) * maxTime)  # utility of slice -- mean for all resources

        ############################  ddpg agent  ###########################################

        tmp_aug_utility_ddpg, tmpx, tmp_real_utility_ddpg = load_and_run_policy(agent_id=agent_id,
                                                                                alpha=alpha[agent_id],
                                                                                weight=weight[agent_id],
                                                                                UENum=UENum, RESNum=RESNum,
                                                                                aug_penalty=z_minus_u)

        x_ddpg = Rmax * np.mean(tmpx, axis=0)  # mean for all maxTime

        aug_utility_ddpg.append(tmp_aug_utility_ddpg)
        real_utility_ddpg.append(tmp_real_utility_ddpg)

        print('iter ' + str(ite))
        print('optimization current allocation is')
        print(x_opt)
        print('ddpg agent current allocation is')
        print(x_ddpg)
        sum_gap.append(np.mean(np.abs(x_ddpg-x_opt)/np.sum(x_opt)))


    end_time = time.time()
    print('Simualtion Time is ' + str(end_time - start_time))

    #####################################          result ploting            ###############################################

    print((np.sum(real_utility_opt) - np.sum(real_utility_ddpg)) / np.sum(real_utility_opt))

    matplt.plot(real_utility_ddpg)
    matplt.plot(real_utility_opt)
    matplt.show()


    with open("saved_test.pickle", "wb") as fileop:
        pickle.dump([x_ddpg, x_opt], fileop)

    scipy.io.savemat('/root/Desktop/DRL_Project/test_agent.mat',
                    mdict={'x_ddpg': x_ddpg,
                    'x_opt': x_opt,
                    'x_static': x_static,
                    'real_utility_ddpg': real_utility_ddpg,
                    'real_utility_opt': real_utility_opt,
                    'real_utility_static': real_utility_static,
                    'sum_gap':sum_gap,})

    print('done')


