import tensorflow as tf
import pdb

class SpeechGender(object):
    def __init__(self, rnn_size = 500,
                       freq_size = 801,
                       max_len = 70,
                       optimizer = 'adam',
                       mode = 'test'):
        self.rnn_size = rnn_size
        self.freq_size = freq_size
        self.max_len = max_len
        self.optimizer = optimizer
        self.mode = mode

        self.input = tf.placeholder(tf.float32, [None, self.max_len, self.freq_size])
        self.seqlen = tf.placeholder(tf.int32, [None])
        self.build_graph()
        if self.mode == 'train':
            self.label = tf.placeholder(tf.float32, [None, 1])
            self.train_op()

    def build_graph(self):
        x = tf.unstack(self.input, self.max_len, 1)
        lstm = tf.contrib.rnn.BasicLSTMCell(self.rnn_size)
        outputs, states = tf.contrib.rnn.static_rnn(lstm, x, sequence_length = self.seqlen,
            dtype=tf.float32)
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
        batch_size = tf.shape(outputs)[0]
        index = tf.range(0, batch_size) * self.max_len + (self.seqlen - 1)
        outputs = tf.gather(tf.reshape(outputs, [-1, self.rnn_size]), index)

        w = tf.get_variable("w", [self.rnn_size, 1])
        b = tf.get_variable("b", [1])
        self.logit = tf.matmul(outputs, w) + b

    def train_op(self):
        # define loss
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels = self.label, 
            logits = self.logit))

        # learning rate
        global_step = tf.Variable(0.0, trainable=False)
        boundaries = [1000.]
        values = [0.00001, 0.00001]
        self.learning_rate = tf.train.piecewise_constant(global_step, boundaries, values)

        # optimizer
        if self.optimizer == 'adam':
            optimizer = tf.train.AdamOptimizer(self.learning_rate)
        else:
            raise NameError("Unknown optimizer type %s" % self.optimizer)

        # training step
        tvars = [var for var in tf.trainable_variables()]
        grads_and_vars = optimizer.compute_gradients(self.loss, tvars)
        self.train_step = optimizer.apply_gradients(grads_and_vars)
