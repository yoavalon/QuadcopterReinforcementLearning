import tensorflow as tf

class Actor():

    def __init__(self, learning_rate=0.001, scope="Actor"):
        with tf.variable_scope(scope):

            self.state = tf.keras.layers.Input(shape=(9,), name="state")
            self.action = tf.keras.layers.Input(shape=(6,), name="action")
            self.target = tf.keras.layers.Input(shape=(1,), name="target")

            inter1 = tf.keras.layers.Dense(
                90,
                name="inter1",
                activation= 'tanh',
                kernel_initializer='random_uniform')(self.state)

            inter2 = tf.keras.layers.Dense(
                90,
                name="inter2",
                activation='tanh',
                kernel_initializer='random_uniform')(inter1)

            inter3 = tf.keras.layers.Dense(
                90,
                name="inter3",
                activation='tanh',
                kernel_initializer='random_uniform')(inter2)

            inter4 = tf.keras.layers.Dense(
                90,
                name="inter4",
                activation='tanh',
                kernel_initializer='random_uniform')(inter3)

            layer_out = tf.keras.layers.Dense(
                6,
                name="actor",
                activation='tanh',
                kernel_initializer='random_uniform')(inter4)


            self.action_out = tf.squeeze(layer_out) # tf.squeeze(tf.nn.softmax(self.output_layer))

            #self.picked_action_prob = tf.gather(self.action_probs, self.action)
            # Loss and train op
            #self.loss = -tf.log(self.picked_action_prob) * self.target

            #CAN'T BE RIGHT !!!!
            self.loss = -tf.log(self.action_out) * self.target

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.action_out, { self.state: state })

    def update(self, state, target, action, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = { self.state: state, self.target: target, self.action: action  }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss
