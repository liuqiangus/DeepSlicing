import numpy as np
import pickle
import sys
import scipy.io
########################################################################################################################
#####################################        Simulation Parameters         #############################################
########################################################################################################################
SliceNum = 3
UENum = 5
RESNum = 1
seed = 12345678
maxTime = 20
ADMM_iter = 10

use_other_utility_function = False  # do no change to True since the function is not convex, cannot be solve by cvxpy

minReward = maxTime * 2
Rmax = 100
Rmin = 0


replay_size = int(1e6)
steps_per_epoch = 5000  # at least 5000 * 100 since we have to learn the augmented state space (ADMM penalty)
epochs = 200
batch_size = 1000
pi_lr = 1e-3
q_lr = 1e-3
hidden_sizes = [128, 128]
start_steps = int(steps_per_epoch*epochs/10)
rho = 10  #################################################################################################################################################################

seed += 1
np.random.seed(seed)
alpha = np.random.uniform(0.1, 0.9, [SliceNum, RESNum, UENum])

seed += 1
np.random.seed(seed)
#weight = np.random.uniform(0.1, 0.9, [SliceNum, UENum])
weight_ue = np.random.uniform(0.1, 0.9, [SliceNum, UENum])

seed += 1
np.random.seed(seed)
weight_slice = np.random.uniform(0.1, 0.9, SliceNum)

weight = np.array([weight_slice[i] * weight_ue[i] for i in range(SliceNum)])


with open("saved_alpha.pickle", "wb") as fileop:
    pickle.dump(alpha, fileop)

with open("saved_weight.pickle", "wb") as fileop:
    pickle.dump(weight, fileop)



#alpha = alpha[[0, 2, 3]]
#weight = weight[[0, 2, 3]]
#SliceNum = 3
# load from training saved parameters

#with open("saved_alpha.pickle", "rb") as fileop:
#    alpha = pickle.load(fileop)
#
#with open("saved_weight.pickle", "rb") as fileop:
#    weight = pickle.load(fileop)
