import tensorflow as tf
import numpy as np
import configs

from Critic import Critic

class PPO(object):

    def __init__(self):
        self.sess = tf.Session()

        self.critic1 = Critic()

        self.tfs = tf.placeholder(tf.float32, [None, configs.S_DIM], 'state')

        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)

        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, configs.A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')

        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                #ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
                ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                surr = ratio * self.tfadv   #IMPORTANT !!!

            self.aloss = -tf.reduce_mean(tf.minimum(
                surr,
                tf.clip_by_value(ratio, 1.-configs.epsilon, 1.+configs.epsilon)*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(configs.A_LR).minimize(self.aloss)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)

        adv = self.critic1.getAdvantage(s,r, self.sess)

        # update actor
        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(configs.A_UPDATE_STEPS)]

        # update critic
        self.critic1.update(s, r, self.sess)

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l4 = tf.layers.dense(self.tfs, 90, tf.nn.tanh, trainable=trainable, kernel_initializer='glorot_normal')
            l3 = tf.layers.dense(l4, 90, tf.nn.tanh, trainable=trainable, kernel_initializer='glorot_normal')
            l2 = tf.layers.dense(l3, 90, tf.nn.tanh, trainable=trainable, kernel_initializer='glorot_normal')
            l1 = tf.layers.dense(l2, 90, tf.nn.tanh, trainable=trainable, kernel_initializer='glorot_normal')
            mu = 2 * tf.layers.dense(l1, configs.A_DIM, tf.nn.tanh, trainable=trainable, kernel_initializer='glorot_normal')
            sigma = tf.nn.sigmoid(tf.layers.dense(l1, configs.A_DIM, tf.nn.softplus, trainable=trainable, kernel_initializer='glorot_normal'))
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)

        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -1, 1)    #was 2

    def getValue(self, s):

        return self.critic1.getValue(s, self.sess)
