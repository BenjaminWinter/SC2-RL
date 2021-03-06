import logging
import multiprocessing as mp
import time

import tensorflow as tf
from absl import flags
from keras import backend as K
from keras.layers import *
from keras.models import *

import a3c.common.shared as shared

FLAGS = flags.FLAGS

flags.DEFINE_float('loss_v', 0.5, 'v loss coefficient')
flags.DEFINE_float('loss_entropy', 0.01, 'entropy coefficient')
flags.DEFINE_integer('min_batch', 32, 'batch Size')


class Brain:
    episodes = 0
    rewards = []
    steps = []
    lock_queue = mp.Lock()
    def __init__(self, s_space, a_space, none_state, saved_model=False, t_queue=None):
        self.logger = logging.getLogger('sc2rl.' + __name__)

        self.s_space = s_space
        self.a_space = a_space
        self.none_state = none_state
        self.queue = t_queue

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

        model = Model(inputs=[c_input], outputs=[out_actions, out_value])
        model._make_predict_function()  # have to initialize before threading

        return model

    def _build_graph(self, model):
        s_t = tf.placeholder(tf.float32, shape=(None, ) + self.s_space)
        a_t = tf.placeholder(tf.float32, shape=(None, self.a_space))
        r_t = tf.placeholder(tf.float32, shape=(None, 1))  # not immediate, but discounted n step reward

        p, v = model(s_t)

        log_prob = tf.log(tf.reduce_sum(p * a_t, axis=1, keep_dims=True) + 1e-10)
        advantage = r_t - v

        loss_policy = - log_prob * tf.stop_gradient(advantage)  # maximize policy
        loss_value = FLAGS.loss_v * tf.square(advantage)  # minimize value error
        entropy = FLAGS.loss_entropy * tf.reduce_sum(p * tf.log(p + 1e-10), axis=1,
                                               keep_dims=True)  # maximize entropy (regularization)

        loss_total = tf.reduce_mean(loss_policy + loss_value + entropy)

        optimizer = tf.train.RMSPropOptimizer(FLAGS.lr, decay=.99)
        minimize = optimizer.minimize(loss_total)

        return s_t, a_t, r_t, minimize

    def optimize(self):
        if self.queue.qsize() < FLAGS.min_batch:
            time.sleep(FLAGS.thread_delay)  # yield
            return
        s = []
        a = []
        r = []
        s_ = []
        s_mask = []

        while not self.queue.empty():
            arr = self.queue.get()

            s.append(arr[0])
            a.append(arr[1])
            r.append(arr[2])
            s_.append(arr[3])
            s_mask.append(arr[4])
        try:
            s = np.stack(s)
            a = np.vstack(a)
            r = np.vstack(r)
            s_ = np.stack(s_)
            s_mask = np.vstack(s_mask)
        except ValueError:
            print('***************************')
            for x in range(len(s)):
                print('s.shape:' + str(s[x].shape))
                print('a:' + str(a[x]))
                print('r:' + str(r[x]))
                print('s_:' + str(s_[x].shape))
                print('s_mask:' + str(s_mask[x]))
                print('________________________')
            print('---------------------------')

        if len(s) > 5 * FLAGS.min_batch:
            self.logger.warning("Optimizer alert! Minimizing batch of %d" % len(s))

        v = self.predict_v(s_)
        r = r + shared.gamma_n * v * s_mask  # set v to 0 where s_ is terminal state

        s_t, a_t, r_t, minimize = self.graph
        self.session.run(minimize, feed_dict={s_t: s, a_t: a, r_t: r})

    def predict(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p, v

    def predict_p(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return p

    def predict_v(self, s):
        with self.default_graph.as_default():
            p, v = self.model.predict(s)
            return v

    def get_episodes(self):
        return self.episodes

    def add_episodes(self, eps):
        self.lock_queue.acquire()
        self.episodes += eps
        self.lock_queue.release()

    def get_rewards(self):
        return self.rewards

    def add_rewards(self, arr):
        self.lock_queue.acquire()
        self.rewards.append(arr)
        self.lock_queue.release()

    def get_steps(self):
        return self.steps

    def add_steps(self, arr):
        self.lock_queue.acquire()
        self.steps.append(arr)
        self.lock_queue.release()

    def save_model(self, str):
        self.model.save(str)

