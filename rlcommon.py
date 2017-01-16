from subprocess import Popen, PIPE
from numpy import uint32
from numpy import int32
from numpy import uint64
from numpy import int64
from time import sleep, time
import random
import os
import socket
import sys

import tensorflow as tf
import tflearn
import numpy as np
import pickle
import redis

# ********** Replay Buffer **************
# Random seed
RANDOM_SEED = 45

# Size of Replay Buffer
BUFFER_SIZE = 100000

MINIBATCH_SIZE = 256



# ********** Tensorflow ******************
# Dimension of state (input)
state_dim = 15

# Dimension of action (output)
action_dim = 1

# Action Bound
action_bound = 0.05

# How long do we save the neural network model
nn_model_save_interval = 10000

ACTOR_LEARNING_RATE = 0.0001#0.00001
CRITIC_LEARNING_RATE = 0.001#0.01#0.001
GAMMA = 0.9  #0.99
TAU = 0 # 0.001


EXPLORATION_RATE = 0.5

SUMMARY_DIR = './results/rl_dpg_train'


# ************ Control *******************
IsRLEnabled = 1 ################### 1

MinimumExecutorUpdateInterval = 0.02 # 50ms

MinimumTrainerUpdateInterval = 0.1 # 100ms


dump_time_interval = 2 # 2 seconds


