"""
@author sourabhxiii

"""
import numpy as np
import tensorflow as tf
from tensorflow import layers
from collections import namedtuple

class Actor:
    def __init__(self, batch_size, input_dim, output_dim, action_low, action_high, np_index, seed):
        tf.set_random_seed(seed)
        self.lr = 1e-2
        bounds = namedtuple("AxnBound", field_names=["low", "high"])
        self.action_bound = bounds(action_low, action_high)
        self.batch_size = batch_size

        with tf.name_scope('input'):
            self.features = tf.placeholder(dtype=tf.float32, shape=(None, input_dim), name='features')
            # self.action = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='action')
            self.grads = tf.placeholder(dtype=tf.float32, shape=(None, output_dim), name='grads')
        with tf.name_scope('output'):
            self.y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='action')


        with tf.name_scope('actor_net'):
            nn = layers.Dense(8, kernel_initializer=tf.random_normal_initializer()
                , bias_initializer=tf.random_normal_initializer()
                , activation=tf.nn.relu, name='AM_D1')(self.features)

            nn = layers.Dense(4, kernel_initializer=tf.random_normal_initializer()
                , bias_initializer=tf.random_normal_initializer()
                , activation=tf.nn.relu, name='AM_D2')(nn)

            # nn = layers.Dense(8, kernel_initializer=tf.random_normal_initializer()
            #     , bias_initializer=tf.random_normal_initializer()
            #     , activation=tf.nn.relu, name='AM_D3')(nn)

            self.axn = layers.Dense(output_dim, kernel_initializer=tf.random_normal_initializer()
                , bias_initializer=tf.random_normal_initializer()
                , activation=tf.nn.tanh, name='AM_D4')(nn)
            # self.predict_op = tf.clip_by_value(self.axn
            #         , np.atleast_2d(self.action_bound.low) 
            #         , np.atleast_2d(self.action_bound.high)
            #         , name='axn_clip'
            #   )
            # self.predict_op = tf.multiply(axn
            #         # , self.action_bound.low
            #         , self.action_bound.high
            #         , name='axn_clip'
            #     )

        self.network_params = tf.trainable_variables()[np_index :]

        with tf.name_scope('actor_gradient'):
            # with tf.name_scope("actor_loss"):
            #     self.loss = tf.reduce_mean(self.axn * self.grads)
            self.unnormalized_actor_gradients = tf.gradients(self.axn, self.network_params, -self.grads)
            # self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        with tf.name_scope("actor_train"):
            self.train_op = tf.train.AdamOptimizer(self.lr).apply_gradients(zip(self.unnormalized_actor_gradients, self.network_params))

        with tf.name_scope("init"):
            self.init = tf.global_variables_initializer()


    def set_session(self, sess):
        self.session = sess
        self.session.run(self.init)

    def model_session(self):
        return self.session

    def get_action(self, obs):
        # print("logits")
        # print(self.session.run(self.logits, feed_dict={self.features: features}))
        # print("axn proba")
        # print(self.session.run(self.action_probs, feed_dict={self.features: features}))
        axn = self.session.run([self.axn], feed_dict={self.features: obs})
        return axn

    def update_weight(self, obs, grads):
        feed_dict = {self.features: obs
            # , self.target_q:  np.atleast_1d(q)
            , self.grads: grads}
        _ = self.session.run([self.train_op], feed_dict)

        # tvars = tf.trainable_variables()
        # tvars_vals = self.session.run(tvars)

        # for var, val in zip(tvars, tvars_vals):
        #     if var.name == 'PM_D4/kernel:0' or var.name == 'VM_D4/kernel:0':
        #         print(var.name, val)  # Prints the name of the variable alongside its value.
        return

    def get_weights(self):
        return self.network_params