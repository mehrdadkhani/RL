import numpy as np
import tensorflow as tf
import tflearn
from replay_buffer import ReplayBuffer
from mahimahiInterface import *







# ==========================
#   Training Parameters
# ==========================
# Max training steps
#MAX_EPISODES = 50000
# Max episode length
#MAX_EP_STEPS = 1000
# Discount factor 
#GAMMA = 0.99
# Soft target update param
#TAU = 0.001

# ===========================
#   Utility Parameters
# ===========================
# Directory for storing tensorboard summary results
#SUMMARY_DIR = './results/rl_dpg_train'
#RANDOM_SEED = 42
# Size of replay buffer
#BUFFER_SIZE = 10000
#MINIBATCH_SIZE = 64

# ===========================
#   Actor and Critic DNNs
# ===========================
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

class ActorNetwork(object):
	""" 
	Input to the network is the state, output is the action
	under a deterministic policy.

	The output layer activation is a sigmoid to keep the action
	between 0 and 1
	"""
	def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau):
		self.sess = sess
		self.s_dim = state_dim
		self.a_dim = action_dim
		self.a_bound = action_bound  # action_bound
		self.learning_rate = learning_rate
		self.tau = tau

		# Actor Network
		self.inputs, self.out, self.scaled_out = self.create_actor_network()

		self.network_params = tf.trainable_variables()


		self.assign_op = []
		self.assign_input = []
		for i in range(len(self.network_params)):
			shape = self.network_params[i].get_shape()
			shape_ = [None,1]
			if len(shape) == 1:
				shape_ = shape
			else:
				shape_ = shape
 
			print(shape_)	
					
			#self.assign_input.append(tflearn.input_data(shape=shape_))
			self.assign_input.append(tf.placeholder(tf.float32, shape=shape_))
			print(self.assign_input[i].get_shape())
			self.assign_op.append(self.network_params[i].assign(self.assign_input[i]))


		# Target Network
		self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()
		
		self.target_network_params = tf.trainable_variables()[len(self.network_params):]

		# Op for periodically updating target network with online network weights
		self.update_target_network_params = \
			[self.target_network_params[i].assign(tf.mul(self.network_params[i], self.tau) + \
				tf.mul(self.target_network_params[i], 1. - self.tau))
				for i in range(len(self.target_network_params))]

		# This gradient will be provided by the critic network
		self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])
		# Combine the gradients here
		self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)
		self.cappedgrad = self.actor_gradients

		for i in range(len(self.actor_gradients)):
			self.cappedgrad[i] = tf.clip_by_average_norm(self.actor_gradients[i], 1.0)
		# self.actor_gradients = [tf.clip_by_average_norm(tf.gradients(self.scaled_out, self.network_params, -self.action_gradient), 100)

		# Optimization Op
		self.optimize = tf.train.RMSPropOptimizer(self.learning_rate).\
			apply_gradients(zip(self.cappedgrad, self.network_params))

		self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

	def assign_params(self,ind, parameter):
		self.sess.run(self.assign_op[ind], feed_dict={
			self.assign_input[ind]: parameter})



	def create_actor_network(self): 
		inputs = tflearn.input_data(shape=[None, self.s_dim])

		# w_init1 = tflearn.initializations.truncated_normal(stddev=1.0/np.sqrt(self.s_dim))
		# layer1 = tflearn.fully_connected(inputs, 1000, activation='relu', weights_init=w_init1)
		# w_init2 = tflearn.initializations.truncated_normal(stddev=1.0/np.sqrt(1000))
		# layer2 = tflearn.fully_connected(layer1, 10, activation='relu', weights_init=w_init2)
		# w_init3 = tflearn.initializations.truncated_normal(stddev=1.0/np.sqrt(10))
		# layer3 = tflearn.fully_connected(layer2, self.a_dim, weights_init=w_init3)
		# out = tflearn.activation(tf.mul(layer2,layer3.W), activation='sigmoid')

		W1 = tf.Variable(tf.truncated_normal([self.s_dim,100], mean=0.0,stddev=1.0/np.sqrt(self.s_dim)))
		b1 = tf.Variable(tf.zeros([100]))
		y1 = tf.nn.relu(tf.matmul(inputs, W1) + b1)

		W2 = tf.Variable(tf.truncated_normal([100,100], mean=0.0, stddev=1.0/np.sqrt(100)))
		b2 = tf.Variable(tf.zeros([100]))
		y2 = tf.nn.relu(tf.matmul(y1, W2) + b2)


		W3 = tf.Variable(tf.truncated_normal([100,10], mean=0.0, stddev=1.0/np.sqrt(100)))
		b3 = tf.Variable(tf.zeros([10]))
		y3 = tf.nn.relu(tf.matmul(y2, W3) + b3)

		W4 = tf.Variable(tf.truncated_normal([10,1], mean=0.0, stddev=1.0/np.sqrt(10)))
		b4 = tf.Variable(tf.constant(0.0,shape=[1]))
		y4 = tf.matmul(y3, W4) + b4

		# b5 = tf.constant(-0.4,shape=[1])
		out = tf.nn.tanh(y4)
		# out = tf.minimum(tf.maximum(y4, y4 * 0.01), 0.01 * (y4-0.8)+0.8)
		scaled_out = tf.mul(out, self.a_bound)
		# scaled_out = out
		return inputs, out, scaled_out


	def train(self, inputs, a_gradient):
		self.sess.run(self.optimize, feed_dict={
			self.inputs: inputs,
			self.action_gradient: a_gradient
		})

	def predict(self, inputs):
		return self.sess.run(self.scaled_out, feed_dict={
			self.inputs: inputs
		})

	def predict_target(self, inputs):
		return self.sess.run(self.target_scaled_out, feed_dict={
			self.target_inputs: inputs
		})

	def update_target_network(self):
		self.sess.run(self.update_target_network_params)

	def get_num_trainable_vars(self):
		return self.num_trainable_vars

	def get_grad_num(self):
		return tf.gradients(self.out, self.network_params)

	def output_gradient(self, inputs, a_gradient):
		return self.sess.run(self.actor_gradients, feed_dict={
			self.inputs: inputs,
			self.action_gradient: a_gradient
		})

	def get_w(self):
		return self.sess.run(self.network_params)



class CriticNetwork(object):
	""" 
	Input to the network is the state and action, output is Q(s,a).
	The action must be obtained from the output of the Actor network.

	"""
	def __init__(self, sess, state_dim, action_dim, learning_rate, tau, num_actor_vars):
		self.sess = sess
		self.s_dim = state_dim
		self.a_dim = action_dim
		self.learning_rate = learning_rate
		self.tau = tau

		# Create the critic network
		self.inputs, self.action, self.out = self.create_critic_network()

		self.network_params = tf.trainable_variables()[num_actor_vars:]

		# Target Network
		self.target_inputs, self.target_action, self.target_out = self.create_critic_network()
		
		self.target_network_params = tf.trainable_variables()[(len(self.network_params) + num_actor_vars):]

		# Op for periodically updating target network with online network weights with regularization
		self.update_target_network_params = \
			[self.target_network_params[i].assign(tf.mul(self.network_params[i], self.tau) + tf.mul(self.target_network_params[i], 1. - self.tau))
				for i in range(len(self.target_network_params))]
	
		# Network target (y_i)
		self.predicted_q_value = tf.placeholder(tf.float32, [None, 1])




		# Define loss and optimization Op
		self.loss = tflearn.mean_square(self.predicted_q_value, self.out)
		self.criticgrads = tf.gradients(self.loss, self.network_params)
		self.cappedgradc = self.criticgrads

		for i in range(len(self.criticgrads)):
			self.cappedgradc[i] = tf.clip_by_average_norm(self.criticgrads[i], 1.0)

		# self.optimize = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)
		self.optimize = tf.train.RMSPropOptimizer(self.learning_rate). \
			apply_gradients(zip(self.cappedgradc, self.network_params))
		# Get the gradient of the net w.r.t. the action
		self.action_grads = tf.gradients(self.out, self.action)


	def create_critic_network(self):
		inputs = tflearn.input_data(shape=[None, self.s_dim])
		action = tflearn.input_data(shape=[None, self.a_dim])

		# w_decay = 1
		# w_init_critic_1 = tflearn.initializations.normal(mean=0, stddev=1.0/np.sqrt(self.s_dim))
		# net1 = tflearn.fully_connected(inputs, 100, activation='tanh', weights_init=w_init_critic_1,regularizer = 'L2', weight_decay = w_decay,bias= True)
		# w_init_critic_2 = tflearn.initializations.normal(mean=0, stddev=1.0/np.sqrt(100))
		# t1 = tflearn.fully_connected(net1, 10, activation='tanh',weights_init=w_init_critic_2,regularizer = 'L2', weight_decay = w_decay,bias= True)
        #
		# w_init_critic_3 = tflearn.initializations.normal(mean=0, stddev=1.0/np.sqrt(self.a_dim))
		# net2 = tflearn.fully_connected(action, 100, activation='tanh', weights_init=w_init_critic_3,regularizer = 'L2', weight_decay = w_decay,bias= True)
		# w_init_critic_4 = tflearn.initializations.normal(mean=0, stddev=1.0/np.sqrt(100))
		# t2 = tflearn.fully_connected(net2, 10, activation='tanh',weights_init=w_init_critic_4,regularizer = 'L2', weight_decay = w_decay,bias= True)
        #
		# net = tflearn.activation(tf.matmul(net1, t1.W) + tf.matmul(net2, t2.W), activation='sigmoid')
        #
		# w_init_critic_5 = tflearn.initializations.normal(mean=0, stddev=1.0/np.sqrt(10))
		# out = tflearn.fully_connected(net, 1, activation='relu', weights_init=w_init_critic_5,regularizer = 'L2', weight_decay = w_decay,bias= True)

		W1c = tf.Variable(tf.truncated_normal([self.s_dim,100], mean=0.0,stddev=1.0/np.sqrt(self.s_dim)))
		b1c = tf.Variable(tf.zeros([100]))
		y1c = tf.nn.relu(tf.matmul(inputs, W1c) + b1c)

		W2c = tf.Variable(tf.truncated_normal([self.a_dim,100], mean=0.0, stddev=1.0/np.sqrt(self.a_dim)))
		b2c = tf.Variable(tf.zeros([100]))
		y2c = tf.nn.relu(tf.matmul(action, W2c) + b2c)


		W3c = tf.Variable(tf.truncated_normal([200,10], mean=0.0, stddev=1.0/np.sqrt(200)))
		b3c = tf.Variable(tf.zeros([10]))
		y3c = tf.nn.relu(tf.matmul(tf.concat(1,[y1c,y2c]), W3c) + b3c)

		W4c = tf.Variable(tf.truncated_normal([10,1], mean=0.0, stddev=1.0/np.sqrt(10)))
		b4c = tf.Variable(tf.constant(0.0,shape=[1]))
		outc = tf.nn.relu(tf.matmul(y3c, W4c) + b4c)

		return inputs, action, outc

	def train(self, inputs, action, predicted_q_value):
		return self.sess.run([self.loss, self.optimize], feed_dict={
			self.inputs: inputs,
			self.action: action,
			self.predicted_q_value: predicted_q_value
		})

	def predict(self, inputs, action):
		return self.sess.run(self.out, feed_dict={
			self.inputs: inputs,
			self.action: action
		})

	def predict_target(self, inputs, action):
		return self.sess.run(self.target_out, feed_dict={
			self.target_inputs: inputs,
			self.target_action: action
		})

	def action_gradients(self, inputs, actions): 
		return self.sess.run(self.action_grads, feed_dict={
			self.inputs: inputs,
			self.action: actions
		})

	def update_target_network(self):
		self.sess.run(self.update_target_network_params)

# ===========================
#   Tensorflow Summary Ops
# ===========================
def build_summaries(): 
	episode_reward = tf.Variable(0.)
	tf.scalar_summary("Reward", episode_reward)
	td_loss = tf.Variable(0.)
	tf.scalar_summary("TD", td_loss)

	summary_vars = [episode_reward, td_loss]
	summary_ops = tf.merge_all_summaries()

	return summary_ops, summary_vars

