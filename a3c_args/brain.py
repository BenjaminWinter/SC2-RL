import threading
import tensorflow as tf
import time
import logging

from keras.layers import *
from keras.models import *
from keras import backend as K

import a3c.shared as shared
import gflags as flags

FLAGS = flags.FLAGS

flags.DEFINE_float('loss_v', 0.5, 'v loss coefficient')
flags.DEFINE_float('loss_entropy', 0.01, 'entropy coefficient')
flags.DEFINE_float('lr', 5e-3, 'learning rate')
flags.DEFINE_integer('min_batch', 32, 'batch Size')


class Brain:
    train_queue = [[], [], [], [], [], [], []]  # s, a,, x, y, r, s', s' terminal mask
    lock_queue = threading.Lock()

    def __init__(self, s_space, a_space, none_state, saved_model=False):
        self.logger = logging.getLogger('sc2rl.' + __name__)

        self.s_space = s_space
        self.a_space = a_space
        self.none_state = none_state
        self.session = tf.Session()
        K.set_session(self.session)
        K.manual_variable_initialization(True)

        if saved_model:
            self.model = load_model(saved_model)
            self.model._make_predict_function()
        else:
            self.model = self._build_model()

        self.graph = self._build_graph(self.model)

        self.session.run(tf.global_variables_initializer())

        if saved_model:
            self.model.load_weights(saved_model)

        self.default_graph = tf.get_default_graph()

        self.default_graph.finalize()  # avoid modifications

    def _build_model(self):

        c_input = Input(shape=self.s_space)
        c1 = Conv2D(
            16,
            kernel_size=8,
            strides=(4, 4),
            activation='relu'
        )(c_input)
        #pool1 = MaxPool2D(pool_size=(2, 2), strides=(1, 1))(c1)
        c2 = Conv2D(
            32,
            kernel_size=4,
            strides=(2, 2),
            activation='relu'
        )(c1)
        #pool2 = MaxPool2D(pool_size=(2, 2), strides=(1, 1))(c2)
        flatten = Flatten()(c2)
        dense1 = Dense(1024, activation='relu')(flatten)
        #dense2 = Dense(50, activation='relu')(dense1)

        out_actions = Dense(self.a_space, activation='softmax')(dense1)
        out_value = Dense(1, activation='linear')(dense1)
        out_x = Dense(FLAGS.screen_resolution, activation='softmax')(dense1)
        out_y = Dense(FLAGS.screen_resolution, activation='softmax')(dense1)

        model = Model(inputs=[c_input], outputs=[out_actions, out_value, out_x, out_y])
        model._make_predict_function()  # have to initialize before threading

        return model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, ) + self.s_space)

        a_t = tf.placeholder(tf.float32, shape=(None, self.a_space))
        x_t = tf.placeholder(tf.float32, shape=(None, FLAGS.screen_resolution))
        y_t = tf.placeholder(tf.float32, shape=(None, FLAGS.screen_resolution))

        r_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward

        p, v, x, y = model(s_t)

        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        log_x = tf.log(tf.reduce_sum(x * x_t, axis=1, keep_dims=True) + 1e-10)
        log_y = tf.log(tf.reduce_sum(y * y_t, axis=1, keep_dims=True) + 1e-10)

        advantage = r_t - v

        loss_policy = - (0.5 * log_prob + 0.25 * log_x + 0.25 * log_y ) * tf.stop_gradient(advantage)  # maximize policy
        loss_value = FLAGS.loss_v * tf.square(advantage)  # minimize value error
        entropy = FLAGS.loss_entropy * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1,
                                               keep_dims=True)  # maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(FLAGS.lr, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, x_t, y_t,  r_t, minimize

    def optimize(self):
        if len(self.train_queue[0]) < FLAGS.min_batch:
            time.sleep(0.01)  # yield
            return

        with self.lock_queue:
            if len(self.train_queue[0]) < FLAGS.min_batch:  # more thread could have passed without lock
                return  # we can't yield inside lock

            s, a, x, y, r, s_, s_mask = self.train_queue

            self.logger.info("Avg Reward: " + str(sum(r)/len(r)))
            self.train_queue = [[], [], [], [], [], [], []]

            s = np.stack(s)
            a = np.vstack(a)
            x = np.vstack(x)
            y = np.vstack(y)
            r = np.vstack(r)
            s_ = np.stack(s_)
            s_mask = np.vstack(s_mask)

        if len(s) > 5 * FLAGS.min_batch:
            self.logger.warning("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + shared.gamma_n * v * s_mask  # set v to 0 where s_ is terminal state

        s_t, a_t, x_t, y_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a,x_t: x, y_t: y, r_t: r})

    def train_push(self, s, a, x, y, r, s_):
        with self.lock_queue:
            self.train_queue[0].append(s)
            self.train_queue[1].append(a)
            self.train_queue[2].append(x)
            self.train_queue[3].append(y)
            self.train_queue[4].append(r)

            if s_ is None:
                self.train_queue[5].append(self.none_state)
                self.train_queue[6].append(0.)
            else:
                self.train_queue[5].append(s_)
                self.train_queue[6].append(1.)

    def predict(self, s):
        with self.default_graph.as_default():
            p, v, x, y = self.model.predict(s)
            return p, v, x, y

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v, x, y = self.model.predict(s)
            return p, x, y

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v, x, y = self.model.predict(s)
            return v
