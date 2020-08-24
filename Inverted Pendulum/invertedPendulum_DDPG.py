'''
Actor - It proposes an action given a state.
Critic - It predicts if the action is good (positive value) 
		or bad (negative value) given a state and an action.
'''

'''
TWO TARGET NETWORK
it add stability to training. In short, we are learning from estimated targets 
and Target networks are updated slowly, hence keeping our estimated targets stable.
'''

'''
We will use Experience Replay(state, action, reward, next_state)
'''

import gym
import tensorflow as tf 
from tensorflow.keras import layers
import numpy as np 
import matplotlib.pyplot as plt 
# from keras.models import load_model


problem = "Pendulum-v0"
env = gym.make(problem)

num_states = env.observation_space.shape[0]
print(f"State Space {num_states}")

num_actions = env.action_space.shape[0]
upper_bound = env.action_space.high[0]
lower_bound = env.action_space.low[0]
print(f"Action Space {num_actions} with bound upper {upper_bound} :: lower {lower_bound}")


'''
Noise Perturbation
The goal of noise perturbation is to expand noise segments to cover unseen scenarios
so that the overfitting problem is mitigated in supervised speech separation.

Ornstein-Uhlenbeck Process
It samples noise from a correlated normal distribution.
'''

class OUActionNoise:
	def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
		self.theta = theta
		self.mean = mean
		self.std_dev = std_deviation
		self.dt = dt
		self.x_initial = x_initial
		self.reset()	

	def __call__(self):
		x = (
			self.x_prev 
			+ self.theta * (self.mean - self.x_prev) * self.dt
			+ self.std_dev * np.sqrt(self.dt)  * np.random.normal(size=self.mean.shape)
		)
		#Uhlenbeck Process Formula, taken from Wiki

		#Store x into x_prev, makes next noise depend on current one
		self.x_prev = x
		return x

	def reset(self):
		if self.x_initial is not None:
			self.x_prev = self.x_initial
		else:
			self.x_prev = np.zeros_like(self.mean)
			# This will copy the shape of self.mean and set values to 0


'''
Critic loss - Mean Squared Error of y - Q(s, a) where y is the expected return 
as seen by the Target network, and Q(s, a) is action value predicted by the Critic 
network. y is a moving target that the critic model tries to achieve; we make this 
target stable by updating the Target model slowly.

Actor loss - This is computed using the mean of the value given by the Critic network
for the actions taken by the Actor network. We seek to MAXIMISE this quantity.
'''


class Buffer:
	def __init__(self, buffer_capacity=100000, batch_size=64):
		# Number of "experiences" to store at max
		self.buffer_capacity = buffer_capacity
		# Num of tuples to train on.
		self.batch_size = batch_size

		# Its tells us num of times record() was called.
		self.buffer_counter = 0

		# Instead of list of tuples as the exp.replay concept go
		# We use different np.arrays for each tuple element
		self.state_buffer = np.zeros((self.buffer_capacity, num_states))
		self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
		self.reward_buffer = np.zeros((self.buffer_capacity, 1))
		self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

	# Takes (s,a,r,s') obervation tuple as input
	def record(self, obs_tuple):
		# Set index to zero if buffer_capacity is exceeded,
		# replacing old records
		index = self.buffer_counter % self.buffer_capacity

		self.state_buffer[index] = obs_tuple[0]
		self.action_buffer[index] = obs_tuple[1]
		self.reward_buffer[index] = obs_tuple[2]
		self.next_state_buffer[index] = obs_tuple[3]

		self.buffer_counter += 1

	# We compute the loss and update parameters
	def learn(self):
		# Get sampling range 
		record_range = min(self.buffer_counter, self.buffer_capacity)
		# Randomly sample indices(batch)
		batch_indices = np.random.choice(record_range, self.batch_size)

		# Convert to tensors
		state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
		action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
		reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
		reward_batch = tf.cast(reward_batch, dtype=tf.float32)
		next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

		# Training and updating Actor & Critic Networks
		# Gradient Tape tracks the automatic differentiation that occurs in a TF model.
		with tf.GradientTape() as tape:
			target_actions = target_actor(next_state_batch)
			y = reward_batch + gamma * target_critic([next_state_batch, target_actions])
			critic_value = critic_model([state_batch, action_batch])
			critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

		critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
		critic_optimizer.apply_gradients(
			zip(critic_grad, critic_model.trainable_variables)
			)

		with tf.GradientTape() as tape:
			actions = actor_model(state_batch)
			critic_value = critic_model([state_batch, actions])
			# Used `-value` as we want to maximize the value given
			# by the critic for our actions
			actor_loss = -tf.math.reduce_mean(critic_value)

		actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
		actor_optimizer.apply_gradients(
			zip(actor_grad, actor_model.trainable_variables)
		)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
def update_target(tau):
	new_weights = []
	target_variables = target_critic.weights
	for i, variable in enumerate(critic_model.weights):
		new_weights.append(variable * tau + target_variables[i] * (1 - tau))

	target_critic.set_weights(new_weights)

	new_weights = []
	target_variables = target_actor.weights
	for i, variable in enumerate(actor_model.weights):
	    new_weights.append(variable * tau + target_variables[i] * (1 - tau))

	target_actor.set_weights(new_weights)


# Now we will define Actor and Critic Networks
'''
These are basic Dense models with ReLU activation. Batch Normalization 
is used to normalize dimensions across samples in a mini-batch, as 
activations can vary a lot due to fluctuating values of input state and action.

----SPECIAL----
We need the initialization for last layer of the Actor to be between -0.003 and 0.003
as this prevents us from getting 1 or -1 output values in the initial stages, 
which would squash our gradients to zero, as we use the tanh activation.
'''
def get_actor():
	# Initialize weights between -3e-3 and 3-e3
	last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

	inputs = layers.Input(shape=(num_states,))
	out1 = layers.Dense(256, activation="relu")(inputs)
	out2 = layers.BatchNormalization()(out1)
	out3 = layers.Dense(256, activation="relu")(out2)
	out4 = layers.BatchNormalization()(out3)
	outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out4)

	# Our upper bound is 2.0 for Pendulum.
	outputs = outputs * upper_bound
	model = tf.keras.Model(inputs, outputs)
	return model

def get_critic():
    # State as input
    state_input = layers.Input(shape=(num_states))
    state_out1 = layers.Dense(8, activation="relu")(state_input)
    state_out2 = layers.BatchNormalization()(state_out1)
    state_out3 = layers.Dense(16, activation="relu")(state_out2)
    state_out4 = layers.BatchNormalization()(state_out3)

    # Action as input
    action_input = layers.Input(shape=(num_actions))
    action_out1 = layers.Dense(16, activation="relu")(action_input)
    action_out2 = layers.BatchNormalization()(action_out1)

    # Both are passed through seperate layer before concatenating
    concat = layers.Concatenate()([state_out4, action_out2])

    out1 = layers.Dense(256, activation="relu")(concat)
    out2 = layers.BatchNormalization()(out1)
    out3 = layers.Dense(256, activation="relu")(out2)
    out4 = layers.BatchNormalization()(out3)
    outputs = layers.Dense(1)(out4)

    # Outputs single value for give state-action
    model = tf.keras.Model([state_input, action_input], outputs)

    return model


# policy() returns an action sampled from our Actor network plus 
# some noise for exploration.
def policy(state, noise_object):
	# .squeeze() function returns a tensor with the same value as its first 
	# argument, but a different shape. It removes dimensions whose size is one. 
    sampled_actions = tf.squeeze(actor_model(state))
    noise = noise_object()
    # Adding noise to action
    sampled_actions = sampled_actions.numpy() + noise

    # We make sure action is within bounds
    # Clip (limit) the values in an array here b/w lower and upper bound
    legal_action = np.clip(sampled_actions, lower_bound, upper_bound)

    return [np.squeeze(legal_action)]


#-------------------HYPER PARAMETERS

std_dev = 0.2
ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

actor_model = get_actor()
critic_model = get_critic()

target_actor = get_actor()
target_critic = get_critic()

'''
# last_actor = load_model('pendulum_actor.h5')
# last_critic = load_model('pendulum_critic.h5')
	Actually we saved weights not model, this would have worked of we do
	model.save('my_model.h5')
'''
# Let us use last saved weights
actor_model.load_weights('pendulum_actor.h5')
critic_model.load_weights('pendulum_critic.h5')

# Making the weights equal initially
target_actor.set_weights(actor_model.get_weights())
target_critic.set_weights(critic_model.get_weights())

# Let us use last saved weights
target_actor.load_weights('pendulum_target_actor.h5')
target_critic.load_weights('pendulum_target_critic.h5')

# Learning rate for actor-critic models
critic_lr = 0.003
actor_lr = 0.001

critic_optimizer = tf.keras.optimizers.Adam(critic_lr)
actor_optimizer = tf.keras.optimizers.Adam(actor_lr)

total_episodes = 30
# Discount factor for future rewards
gamma = 0.995
# Used to update target networks
tau = 0.005

buffer = Buffer(40000, 64)


#------------------------------------TRAINING 

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

# Takes about 20 min to train
for ep in range(total_episodes):

    prev_state = env.reset()
    episodic_reward = 0

    while True:
        # To see the Actor in action
        env.render()

        tf_prev_state = tf.expand_dims(tf.convert_to_tensor(prev_state), 0)

        action = policy(tf_prev_state, ou_noise)
        # Recieve state and reward from environment.
        state, reward, done, info = env.step(action)

        buffer.record((prev_state, action, reward, state))
        episodic_reward += reward

        buffer.learn()
        update_target(tau)

        # End this episode when `done` is True
        if done:
            break

        prev_state = state

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

# Plotting graph
# Episodes versus Avg. Rewards
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()


# Save the weights
actor_model.save_weights("pendulum_actor.h5")
critic_model.save_weights("pendulum_critic.h5")

target_actor.save_weights("pendulum_target_actor.h5")
target_critic.save_weights("pendulum_target_critic.h5")