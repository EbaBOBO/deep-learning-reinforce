import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class Reinforce(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The Reinforce class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this, 
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(Reinforce, self).__init__()
        self.num_actions = num_actions
        
        # TODO: Define network parameters and optimizer
        self.num_input = state_size
        self.dense_layer1 = tf.keras.layers.Dense(100, activation= 'relu')
        self.dense_layer2 = tf.keras.layers.Dense(self.num_actions)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        

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
        # TODO: implement this ~
        # print('call')
        states1 = self.dense_layer1(states)
        states2 = self.dense_layer2(states1)
        prob = tf.nn.softmax(states2)
    
        return prob

        pass

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Make sure to understand the handout clearly when implementing this.

        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a Tensorflow scalar
        """
        # TODO: implement this
        # Hint: Use gather_nd to get the probability of each action that was actually taken in the episode.
        # print('loss,states shape',states.shape,actions)
        actions = list(actions)
        action = tf.convert_to_tensor([[i,num] for i,num in enumerate(actions)])

        pro = self.call(states)
        pro1 = tf.gather_nd(pro, action)
        # print(pro1)

        discounted_rewards = tf.reshape(discounted_rewards,[1,-1])
        # print(discounted_rewards)
        # print(pro1.shape,discounted_rewards.shape)
        # print('log',tf.math.log(pro1))
        logits = tf.reduce_sum(-tf.math.log(pro1)*discounted_rewards)
        # print(logits)
        return logits
        pass

