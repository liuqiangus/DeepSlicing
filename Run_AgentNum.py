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
from ADMM import admm_mix_algorithm

if __name__ == "__main__":

    simulated_optimization = [True] * SliceNum
    utility = np.zeros(SliceNum)

    for i in range(SliceNum):

        for j in range(i):

            simulated_optimization[j] = False

        utility[i] = admm_mix_algorithm(SliceNum, UENum, RESNum, alpha, weight, simulated_optimization)[-1]

    scipy.io.savemat('/root/Desktop/DRL_Project/result_agent_num.mat', mdict={'utility': utility,})

