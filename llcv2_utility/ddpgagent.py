"""
@author sourabhxiii

"""
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import random
import numpy as np
from replaybuffer import ReplayBuffer
from ounoise import OUNoise
from actormodel import Actor
from criticmodel import Critic
import tensorflow as tf

BUFFER_SIZE = int(1e5)
BATCH_SIZE = 64
BATCH_SIZE = MINI_BATCH_SIZE = 32
GAMMA = 0.97
TAU = 1e-3

class Agent:
    """
    An agent.
    """
    def __init__(self, state_size, axn_size, axn_low, axn_high, sess, seed):
        # . inputs
        self.state_size = state_size
        self.action_size = axn_size
        self.action_low = axn_low
        self.action_high = axn_high
        self.sess = sess
        self.gamma = GAMMA
        self.seed = random.seed(seed)
        self.buffer_size = BUFFER_SIZE
        self.batch_size = BATCH_SIZE
        self.mini_batch_size = MINI_BATCH_SIZE

        # . memory
        self.memory = ReplayBuffer(self.buffer_size, self.mini_batch_size, self.seed)
        # . noise process
        self.noise = OUNoise(self.action_size, self.seed)

        np_index = 0
        # . Actor network
        self.actor = Actor(self.mini_batch_size, self.state_size, self.action_size
                , self.action_low, self.action_high, np_index, self.seed)
        np_index += len(self.actor.get_weights())
        # . Critic network
        self.critic = Critic(self.state_size, axn_size, 1, np_index, self.seed)
        np_index += len(self.critic.get_weights())
        # . Target network
        self.actor_target = Actor(self.mini_batch_size, self.state_size, self.action_size
                , self.action_low, self.action_high, np_index, self.seed)
        np_index += len(self.actor_target.get_weights())
        self.critic_target = Critic(self.state_size, axn_size, 1, np_index, self.seed)


    
    def reset(self):
        """ Resets the memory and noise process """
        # . reset memory
        self.memory.reset()
        # . reset the noise proc
        self.noise.reset()
        return
    def set_session(self):
        # . set session
        self.actor.set_session(self.sess)
        self.critic.set_session(self.sess)
        self.actor_target.set_session(self.sess)
        self.critic_target.set_session(self.sess)
        return

    def act(self, state):
        """
        Returns the action for the given state.
        """
        # . get the action
        axn = self.actor.get_action(np.atleast_2d(state))
        # . add OU noise to it
        axn += self.noise.sample()
        axn = np.squeeze(np.clip(axn, self.action_low, self.action_high))
        return axn

    def step(self, state, axn, reward, state_1, done):
        """
        Saves experiences into memory until there are enough experince to learn from.
        """
        # . save experience
        self.memory.add_experience(state, axn, reward, state_1, done)
        # . learn if enough experience has been accumulated
        if self.memory.get_length() >= self.memory.batch_size:
            # state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            #     self.memory.get_batch()
            self.learn(self.memory.get_batch())
        return

    def learn(self, experience_batch):
        """
        Updates Actor, Critic and Target weights using the given batch of experiences.
        """
        """ unpack the collection """
        states, actions, rewards, next_states, dones = experience_batch
        
        """ update critic """
        # . get target_axns for next_states
        axn_next_states = self.actor_target.get_action(next_states)
        # . get target Q (next_states, target_axns)
        target_Qs_next = self.critic_target.get_q(next_states, axn_next_states)
        # . get target Q for current state (y value)
        target_ys = rewards.reshape(self.mini_batch_size, 1) + self.gamma * target_Qs_next * (1- dones.reshape(self.mini_batch_size, 1))
        # # . get predicted Qs
        # predicted_Qs = self.critic.get_q(states)
        # . update critic weight
        self.critic.update_weight(states, actions, target_ys)

        """ update actor """
        # . get predicted action
        predicted_axns = self.actor.get_action(states)

        grads = self.critic.get_action_gradients(states, predicted_axns)
        # # . get Q value from critic for the state and the selected axn
        # target_Qs = self.critic.get_q(states, predicted_axns)
        # # . calculate loss
        # actor_loss = - target_Qs
        # . update actor weight
        self.actor.update_weight(states, grads[0])

        """ update target """
        self.soft_update(self.critic, self.critic_target, TAU)
        self.soft_update(self.actor, self.actor_target, TAU)
        return


    def soft_update(self, base_model, target_model, tau):
        base_weights = base_model.get_weights()
        target_weights = target_model.get_weights()
        # for b_tvar, t_tvar in zip(base_weights, target_weights):
        #     print(b_tvar, t_tvar)
        # for i in range(len(base_weights)):
        #     target_weights[i] = tau * base_weights[i] + (1 - tau)* target_weights[i]
        self.update_target_weights = [target_weights[i].assign(tf.multiply(base_weights[i], tau) +
                                                  tf.multiply(target_weights[i], 1. - tau))
                for i in range(len(base_weights))]
        self.sess.run(self.update_target_weights)
        return