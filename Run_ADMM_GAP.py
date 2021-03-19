import cvxpy as cp
import matplotlib.pyplot as matplt
from utils import *
from test_ddpg import *
from ddpg_alg_spinup import ddpg
import tensorflow as tf
from env_mra import ResourceEnv
import numpy as np
import time
import pickle
import scipy.io
from parameters import *
from functions import *
from ADMM import admm_ddpg_algorithm
from ADMM import admm_opt_algorithm
from ADMM import admm_static_algorithm


if __name__ == "__main__":

    #utility = np.zeros(ADMM_iter)
    INDEX = np.arange(SliceNum)

    utility_static, gap_static = admm_static_algorithm(SliceNum, UENum, RESNum, alpha, weight)
    print(utility_static)

    utility, gap = admm_ddpg_algorithm(SliceNum, UENum, RESNum, alpha, weight, INDEX)
    print(utility)

    utility_opt, gap_opt = admm_opt_algorithm(SliceNum, UENum, RESNum, alpha, weight)
    print(utility_opt)

    scipy.io.savemat('/root/Desktop/DRL_Project/result_ADMM_GAP.mat', mdict={'utility': utility, 'utility_opt': utility_opt, 'utility_static': utility_static,
                                                                             'gap': gap, 'gap_opt': gap_opt, 'gap_static': gap_static,})

