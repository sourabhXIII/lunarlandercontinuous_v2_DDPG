"""
@author sourabhxiii

"""
import numpy as np
import tensorflow as tf
from tensorflow import layers
from collections import namedtuple

class Critic:
    def __init__(self, state_dim, axn_dim, output_dim, np_index, seed):
        tf.set_random_seed(seed)
        self.lr = 3e-2

        with tf.name_scope('input'):
            self.features = tf.placeholder(dtype=tf.float32, shape=(None, state_dim), name='features')
            self.action = tf.placeholder(dtype=tf.float32, shape=(None, axn_dim), name='action')
            self.target_q = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='target_q')
        with tf.name_scope('output'):
            self.y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='qvalue')


        with tf.name_scope('critic_net'):
            nn = layers.Dense(8, kernel_initializer=tf.random_normal_initializer()
                , bias_initializer=tf.random_normal_initializer()
                , activation=tf.nn.relu, name='CM_D1')(self.features)

            nn = tf.concat([layers.flatten(nn), self.action], 1)

            nn = layers.Dense(8, kernel_initializer=tf.random_normal_initializer()
                , bias_initializer=tf.random_normal_initializer()
                , activation=tf.nn.relu, name='CM_D2')(nn)

            nn = layers.Dense(4, kernel_initializer=tf.random_normal_initializer()
                , bias_initializer=tf.random_normal_initializer()
                , activation=tf.nn.relu, name='CM_D3')(nn)

            self.predict_op = layers.Dense(output_dim, kernel_initializer=tf.random_normal_initializer()
                , bias_initializer=tf.random_normal_initializer()
                , activation=tf.nn.tanh, name='CM_D4')(nn)
        
        with tf.name_scope('critic_gradient'):
            with tf.name_scope("critic_loss"):
                self.loss = tf.reduce_mean(
                                tf.losses.mean_squared_error(labels=self.target_q
                                        , predictions=self.predict_op))
            with tf.name_scope("critic_train"):
                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        """
        https://pemami4911.github.io/blog/2016/08/21/ddpg-rl.html
        https://github.com/pemami4911/deep-rl/tree/master/ddpg
        """
        # Get the gradient of the net w.r.t. the action.
        # For each action in the minibatch (i.e., for each x in xs),
        # this will sum up the gradients of each critic output in the minibatch
        # w.r.t. that action. Each output is independent of all
        # actions except for one.
        self.action_grads = tf.gradients(self.predict_op, self.action)
        
        with tf.name_scope("init"):
            self.init = tf.global_variables_initializer()

        self.network_params = tf.trainable_variables()[np_index :]
        return

    def set_session(self, sess):
        self.session = sess
        self.session.run(self.init)

    def model_session(self):
        return self.session

    def update_weight(self, obs, axn, target):
        self.session.run(self.train_op
                , feed_dict={self.features: obs
                    , self.action: axn
                    , self.target_q: target})
        
    def get_q(self, obs, axns):
        return self.session.run(self.predict_op
                , feed_dict={self.features: obs
                , self.action: np.squeeze(axns)})

    def get_action_gradients(self, obs, axns):
        return self.session.run(self.action_grads
                , feed_dict={self.features: obs
                , self.action: np.squeeze(axns)})
        
    def get_weights(self):
        return self.network_params
