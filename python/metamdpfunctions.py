from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import inspect
import os

from tf_agents.environments import utils
from tf_agents.agents.ppo import ppo_agent
from tf_agents.agents.reinforce import reinforce_agent
from tf_agents.environments import py_environment
from tf_agents.environments import tf_py_environment
from tf_agents.networks import actor_distribution_network
from tf_agents.networks import value_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.trajectories import trajectory
from tf_agents.trajectories import time_step as ts
from tf_agents.specs import array_spec
from tf_agents.networks import normal_projection_network
from tf_agents.policies import policy_saver

#for testing
from tf_agents.utils import nest_utils
from tf_agents.specs import distribution_spec
from tf_agents.agents.ppo import ppo_utils

tf.compat.v1.enable_v2_behavior()

def compute_avg_return(environment, policy, num_episodes=10):
	"""
	Estimates expected future rewards from a the starting state of a given environment under a policy
	by simulating episodes and averaging empirical rewards
	"""
	total_return = 0.0
	total_length = 0
	for _ in range(num_episodes):

		time_step = environment.reset()
		episode_return = 0.0

		while not time_step.is_last():
			action_step = policy.action(time_step)
			time_step = environment.step(action_step.action)
			episode_return += time_step.reward
			total_length += 1

		total_return += episode_return

	avg_return = total_return / num_episodes
	return avg_return.numpy()[0],total_length/num_episodes

def collect_episode(replay_buffer, environment, policy, num_episodes,verbose):
	"""
	Simulates a number of epsiodes in the environment with the given policy and
	adds all steps (not episodes) to the replay buffer.
	"""
	episode_counter = 0
	environment.reset()

	while episode_counter < num_episodes:
		time_step = environment.current_time_step()
		action_step = policy.action(time_step)
		next_time_step = environment.step(action_step.action)
		traj = trajectory.from_transition(time_step, action_step, next_time_step)

		# Add trajectory to the replay buffer
		replay_buffer.add_batch(traj)

		if traj.is_boundary():
			episode_counter += 1 #Do not need to manually reset here because environment.step() does that
			if verbose:
				print(episode_counter)

def load_ppo_agent(train_env,actor_fc_layers,value_fc_layers,learning_rate,num_epochs,preprocessing_layers=None,preprocessing_combiner=None):
	"""
	Function which creates a tensorflow agent for a given environment with specified parameters, which uses the 
	proximal policy optimization (PPO) algorithm for training. 
	actor_fc_layers: tuple of integers, indicating the number of units in intermediate layers of the actor network. All layers are Keras Dense layers
	actor_fc_layers: same for value network
	preprocessing_layers: already-contructed layers of the preprocessing networks, which converts observations to tensors. Needed when the observation spec is either a list or dictionary
	preprocessing_combiner: combiner for the preprocessing networks, typically by concatenation. 
	learning_rate: learning rate, recommended value 0.001 or less
	num_epochs: number of training epochs which the agent executes per batch of collected episodes. 
	
	For more details on PPO, see the documentation of tf_agents: https://github.com/tensorflow/agents/tree/master/tf_agents
	or the paper: https://arxiv.org/abs/1707.06347
	"""

	optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate) #using Adam, a learning rule which uses only first-order gradients but incorporates momentum to become approximately second-order 
	
	train_step_counter = tf.compat.v2.Variable(0) #this creates a counter that starts at 0

	actor_net = actor_distribution_network.ActorDistributionNetwork(
		train_env.observation_spec(),
		train_env.action_spec(),
		preprocessing_combiner=preprocessing_combiner,
		preprocessing_layers=preprocessing_layers,
		fc_layer_params=actor_fc_layers,
	)
	value_net = value_network.ValueNetwork(
		train_env.observation_spec(),
		preprocessing_combiner=preprocessing_combiner,
		preprocessing_layers=preprocessing_layers,
		fc_layer_params=value_fc_layers
	)

	tf_agent = ppo_agent.PPOAgent(
		train_env.time_step_spec(),
		train_env.action_spec(),
		optimizer=optimizer,
		actor_net=actor_net,
		value_net=value_net,
		num_epochs=num_epochs,
		train_step_counter=train_step_counter,
		normalize_rewards=False, #This is crucial to avoid the agent geting stuck
		normalize_observations=False, #same
		discount_factor=1.0,
	)

	tf_agent.initialize() #This is necessary to create variables for the networks
	return tf_agent


#Note: the function below is copied from https://github.com/tensorflow/agents/blob/master/tf_agents/agents/ppo/ppo_utils.py
#But that code contains a bug (or at least it is incompatible with our environment)
#which this function fixes. The problem is that the code as written in ppo_utils crashes when 
#the nested_from_distribution and nested_to_distribution are lists of distributions over actions, 
#and all actions have different size. This code solves that by introducing a reduce_sum operator at a strategically chosen point
def nested_kl_divergence_new(nested_from_distribution, nested_to_distribution,
												 outer_dims=()):
	
	"""Given two nested distributions, sum the KL divergences of the leaves."""
	

	tf.nest.assert_same_structure(nested_from_distribution,nested_to_distribution)

	# Make list pairs of leaf distributions.
	flat_from_distribution = tf.nest.flatten(nested_from_distribution)
	flat_to_distribution = tf.nest.flatten(nested_to_distribution)
		

	all_kl_divergences = [tf.reduce_sum(from_dist.kl_divergence(to_dist),axis=2)
						  for from_dist, to_dist in zip(flat_from_distribution, flat_to_distribution)]
	#this line is the only one that is changed from ppo_utils.py
	#previously, it read
	#all_kl_divergences = [from_dist.kl_divergence(to_dist)
	#					  for from_dist, to_dist in zip(flat_from_distribution, flat_to_distribution)]		
	#The reduce_sum operation is necessary to avoid shape mismatch errors

	# Sum the kl of the leaves.
	summed_kl_divergences = tf.add_n(all_kl_divergences)

	# Reduce_sum over non-batch dimensions.
	reduce_dims = list(range(len(summed_kl_divergences.shape)))
	for dim in outer_dims:
		reduce_dims.remove(dim)
	total_kl = tf.reduce_sum(input_tensor=summed_kl_divergences, axis=reduce_dims)

	return total_kl
	
def test_environment(py_env,observe_action,terminate_action):
	"""
	Helper function which tests out a metamdp environment. If this runs without crashing, it
	is likely that the environment does not contain egregious bugs, at least the inputs/outputs 
	are likely to match the required action and observation specs. Of course, the transition logic of the 
	environment may still be messed up.
	"""
	print('ObservationSpec:',py_env.observation_spec())
	print('ActionSpec:',py_env.action_spec())
	
	time_step = py_env.reset()
	cumulative_reward = time_step.reward
	print(cumulative_reward)

	for a in [observe_action]*10+[terminate_action]:
		time_step = py_env.step(a)
		cumulative_reward += time_step.reward
		print(cumulative_reward)
	#these lines compute the reward on a single episodes where the agent takes an `observe' action 10 times, then terminates
	
	print('Final Reward = ', cumulative_reward)
	utils.validate_py_environment(py_env)
	#this is a validation function provided by tf_agents, which does something similar to the code above but also checks 
	#for spec mismatches more explicitly

def load_reinforce_agent(train_env,actor_fc_layers,learning_rate,num_epochs,preprocessing_layers=None,preprocessing_combiner=None):
	"""
	Creates a REINFORCE agent, using the same inputs as load_ppo_agent. Note that the reinforce algorithm is pure policy gradient and does not have an critic (i.e., value) network. 
	"""

	optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
	train_step_counter = tf.compat.v2.Variable(0)

	actor_net = actor_distribution_network.ActorDistributionNetwork(
		train_env.observation_spec(),
		train_env.action_spec(),
		fc_layer_params=actor_fc_layers,
		preprocessing_layers=preprocessing_layers,
		preprocessing_combiner=preprocessing_combiner
	)

	tf_agent = reinforce_agent.ReinforceAgent(
		train_env.time_step_spec(),
		train_env.action_spec(),
		actor_network=actor_net,
		optimizer=optimizer,
		normalize_returns=False,
		train_step_counter=train_step_counter)

	tf_agent.initialize()
	return tf_agent

def train_agent(tf_agent,train_env,eval_env,num_iterations,returns,losses,
				collect_episodes_per_iteration,log_interval,eval_interval,policy_checkpoint_interval,
				replay_buffer_capacity,num_eval_episodes,direc,verbose=False):

	"""
	Main training function, which optimizes an agent in a given metamdp environment
	inputs:
	tf_agent: either a REINFORCE or PPO agent, as created by load_reinforce_agent or load_ppo_agent. 
	Note that train_agent does not require num_epochs or actor/value/preprocessing layers as arguments, since 
	they are implicitly provided through tf_agent
	train_env: environment in which the agent will collect episodes to train on
	eval_env: environment in which the agent will collect episodes to monitor its performance (i.e., learning curve). It is apparently good practice to separate these environments even though they are both instances of the same object with the same settings. 
	Note that both train_env and eval_env should not be a MetaMDPEnv, but instead a tf_py_environment.TFPyEnvironment(), 
	which is a function that tf_agents provided to convert actions/observations in a gym environment to tensors. 
	returns, losses: arrays to which the return and loss will be appended at regular intervals. Generally I train with returns = [] and losses = []
	collect_episodes_per_iteration: number of episodes to collect per training iteration. Note that the algorithm will train on this data for multiple epochs.
	log_interval: how often the training algorithm should report the training loss. Note that this interval should be provided in training steps, which increments by num_epochs after each call to tf_agent.train(). Therefore, log_interval should be an integer multiple of num_epochs. 
	eval_interval: same for evaluation of expected returns. 
	policy_checkpoint_interval: same for regular dumps of the policy parameters
	replay_buffer_capacity: size if the buffer that train_agent stores episodes in. Note that the size is measured in steps, not episodes. For example, if replay_buffer_capacity is 1000 and each episode is 10 steps, this buffer will store 100 episodes. 
	num_eval_episodes: number of episodes to collect when evaluating the expected return. More episodes imply more accurate estimates but also take more time
	direc: directory in which output will be saved. If the directory does not exist, it will be created
	verbose: flag which controls the amount of output, default False
	"""

	if not os.path.exists(direc):
		os.mkdir(direc)
		
	with open(os.path.join(direc,'results.txt'),'w',buffering=1) as f: #buffering=1 causes the file to be flushed after every line
		print(inspect.getargvalues(inspect.currentframe()),file=f) # this logs all argument values the log file f

		replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
			data_spec=tf_agent.collect_data_spec,
			batch_size=train_env.batch_size,
			max_length=replay_buffer_capacity
		)
		#initializing the replay buffer
		
		if not policy_checkpoint_interval is None:
			policy_checkpointer = common.Checkpointer(
				ckpt_dir=os.path.join(direc, 'policies/'),
				policy=tf_agent.collect_policy)
			saved_model = policy_saver.PolicySaver(
				tf_agent.collect_policy)
			#initializing the objects needed to create regular dumps of the policy parameters

		tf_agent.train = common.function(tf_agent.train)
		
		# Set the training step counter to 0
		tf_agent.train_step_counter.assign(0)
		
		for _ in range(num_iterations):
			# Collect a few episodes using collect_policy and save to the replay buffer.
			if verbose:
				print("collecting " + str(collect_episodes_per_iteration) + " episodes")
			collect_episode(replay_buffer, train_env, tf_agent.collect_policy, collect_episodes_per_iteration,verbose)
			trajectories = replay_buffer.gather_all()
			# Use data from the buffer and update the agent's network.
			
			step = tf_agent.train_step_counter.numpy()

			if step % eval_interval == 0:
				avg_return,avg_length = compute_avg_return(eval_env, tf_agent.collect_policy, num_eval_episodes)
				m = 'step = {0}: Average Collection Return = {1}'.format(step, avg_return)
				print(m)
				print(m,file=f)
				m = 'step = {0}: Average Collection Length = {1}'.format(step, avg_length)
				print(m)
				print(m,file=f)
				returns.append(avg_return)
				#log the average return and the average episode length (that is, the average number of observations that the agent takes before terminating
				
			if not policy_checkpoint_interval is None and step % policy_checkpoint_interval == 0:
			  policy_checkpointer.save(global_step=step)
			  saved_model_path = os.path.join(direc, 'policies/policy_' + ('%d' % step).zfill(9))
			  saved_model.save(saved_model_path)
			
			if verbose:
				print("training")
			train_loss = tf_agent.train(experience=trajectories)
			#Note that usually one would write 
			#replay_buffer.clear()
			#to clear the buffer of episodes taken before this training step, but we keep it to 
			#keep a rolling buffer of episodes. When adding to the buffer it automatically kicks older epsiodes out (afaik)
			
			if step % log_interval == 0:
				m = 'step = {0}: loss = {1}'.format(step, train_loss.loss)
				print(m)
				print(m,file=f)
				losses.append(train_loss.loss.numpy())


		return tf_agent,returns,losses