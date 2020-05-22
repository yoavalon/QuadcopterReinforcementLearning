import tensorflow as tf

class Actor():

    def __init__(self, learning_rate=0.0005, scope="Actor"):
        with tf.variable_scope(scope):

            self.state = tf.keras.layers.Input(shape=(9,), name="state")
            self.action = tf.keras.layers.Input(shape=(6,), name="action")
            self.advantage = tf.keras.layers.Input(shape=(1,), name="advantage")  #was target

            inter1 = tf.keras.layers.Dense(
                18,
                name="inter1",
                activation= 'tanh',
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None))(self.state)

            inter2 = tf.keras.layers.Dense(
                18,
                name="inter2",
                activation='tanh',
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None))(inter1)

            '''
            inter3 = tf.keras.layers.Dense(
                90,
                name="inter3",
                activation='tanh',
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))(inter2)

            inter4 = tf.keras.layers.Dense(
                90,
                name="inter4",
                activation='tanh',
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))(inter3)
            '''

            layer_out = tf.keras.layers.Dense(
                6,
                name="actor",
                activation= None, #'tanh',
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=None))(inter2)

            self.action_out = tf.squeeze(layer_out) # tf.squeeze(tf.nn.softmax(self.output_layer))

            #CAN'T BE RIGHT !!!! -> target here should be advantage
            #self.loss = -tf.log(self.action_out) * self.advantage
            self.loss = -tf.log(tf.abs(self.action_out)) * self.advantage

            #In this continous case we get nans in the loos since
            #the action_output can be negative, then log(-x) is nan like 1/0
            #This loss is made for discrete actions.
            #Solution: Convert to discrete action.. simple as that

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())


    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_out, { self.state: state })

    def update(self, state, advantage, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.advantage: advantage, self.action: action  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss
