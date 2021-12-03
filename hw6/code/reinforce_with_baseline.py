import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class ReinforceWithBaseline(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.

        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(ReinforceWithBaseline, self).__init__()
        self.num_actions = num_actions

        # TODO: Define actor network parameters, critic network parameters, and optimizer
        self.num_input = state_size
        self.hidden_layer = 100
        self.dense_layer1 = tf.keras.layers.Dense(self.hidden_layer, activation= 'relu')
        self.dense_layer2 = tf.keras.layers.Dense(self.num_actions)
        self.critic_layer1 = tf.keras.layers.Dense(self.hidden_layer, activation= 'relu')
        self.critic_layer2 = tf.keras.layers.Dense(1)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        pass

    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        for each state in the episode
        """
        # TODO: implement this!
        states1 = self.dense_layer1(states)
        states2 = self.dense_layer2(states1)
        prob = tf.nn.softmax(states2)
    
        return prob
        pass

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An [episode_length, state_size] dimensioned array representing the history of states
        of an episode.
        :return: A [episode_length] matrix representing the value of each state.
        """
        # TODO: implement this :D
        states1 = self.critic_layer1(states)
        prob = self.critic_layer2(states1)
    
        return prob
        pass

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Refer to the lecture slides referenced in the handout to see how this is done.

        Remember that the loss is similar to the loss as in part 1, with a few specific changes.

        1) In your actor loss, instead of element-wise multiplying with discounted_rewards, you want to element-wise multiply with your advantage. 
        See handout/slides for definition of advantage.
        
        2) In your actor loss, you must use tf.stop_gradient on the advantage to stop the loss calculated on the actor network 
        from propagating back to the critic network.
        
        3) See handout/slides for how to calculate the loss for your critic network.

        :param states: A batch of states of shape (episode_length, state_size)
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """
        # TODO: implement this :)
        # Hint: use tf.gather_nd (https://www.tensorflow.org/api_docs/python/tf/gather_nd) to get the probabilities of the actions taken by the model
        value = self.value_function(states)
        advantage = discounted_rewards - value
        l_critic = tf.square(advantage)

        actions = list(actions)
        action = tf.convert_to_tensor([[i,num] for i,num in enumerate(actions)])
        pro = self.call(states)
        pro1 = tf.gather_nd(pro, action)

        discounted_rewards = tf.reshape(discounted_rewards,[1,-1])

        l_actor = tf.reduce_sum(-tf.math.log(pro1)*tf.stop_gradient(advantage))

        logits = l_actor + l_critic

        return logits
        pass
