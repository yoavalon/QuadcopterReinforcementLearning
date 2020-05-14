import tensorflow as tf

class Critic():

    def __init__(self, learning_rate=0.0008, scope="Critic"):
        with tf.variable_scope(scope):

            self.state = tf.keras.layers.Input(shape=(9,), name="state")
            self.target = tf.keras.layers.Input(shape=(1,), name="target")

            CriticInter1 = tf.keras.layers.Dense(
                18,
                name="CriticInter1",
                activation= 'tanh',
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))(self.state)

            self.output_layer = tf.keras.layers.Dense(
                1,
                name="critic",
                activation=None,
                kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=None))(CriticInter1)

            #may not be necessary
            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        return sess.run(self.value_estimate, { self.state: state })

    def update(self, state, target, sess=None):
        sess = sess or tf.get_default_session()
        feed_dict = {self.state: state, self.target: target }
        _, loss = sess.run([self.train_op, self.loss], feed_dict)
        return loss

    #interesting
    def getTableMean(self, sess=None, scope = None) :
        sess = sess or tf.get_default_session()
        gr = tf.get_default_graph()
        var = tf.reduce_mean(gr.get_tensor_by_name(f'{scope}/fully_connected/weights:0')).eval()
        var2 = tf.reduce_mean(gr.get_tensor_by_name(f'{scope}/fully_connected/biases:0')).eval()
        return var, var2
