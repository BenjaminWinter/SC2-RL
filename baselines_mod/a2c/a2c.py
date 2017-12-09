#!/usr/bin/env python
import os, logging, gym, time
import os.path as osp
from baselines import logger
from baselines.common import set_global_seeds, explained_variance
from baselines import bench
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from absl import flags
import util.helpers as helpers
import tensorflow as tf

FLAGS = flags.FLAGS

if FLAGS.action_args:
    from baselines_mod.a2c.a2c_disc import Runner, Model
    from baselines_mod.a2c.policies_coords import CnnPolicy, LstmPolicy, LnLstmPolicy
else:
    from baselines.a2c.a2c import Runner, Model
    from baselines_mod.a2c.policies import CnnPolicy, LstmPolicy, LnLstmPolicy

flags.DEFINE_enum('policy', 'cnn', ['cnn', 'lstm', 'lnlstm'], 'Policy architecture')
flags.DEFINE_enum('lrschedule', 'constant',['linear', 'constant'], 'Learning rate schedule')
flags.DEFINE_integer('seed', 0, 'RNG seed')


class A2c:
    def __init__(self, run_time, a_space, s_space):
        pass

    def run(self):
        self.train(num_frames=FLAGS.run_time, seed=FLAGS.seed,
              policy=FLAGS.policy, lrschedule=FLAGS.lrschedule, num_cpu=FLAGS.threads)

    def train(self, num_frames, seed, policy, lrschedule, num_cpu):

        def make_env(rank):
            def _thunk():
                env = helpers.get_env_wrapper(render=FLAGS.render)
                env.seed(seed + rank)
                env = bench.Monitor(env, logger.get_dir() and
                                    os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
                gym.logger.setLevel(logging.WARN)
                return env

            return _thunk

        set_global_seeds(seed)
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
        if policy == 'cnn':
            policy_fn = CnnPolicy
        elif policy == 'lstm':
            policy_fn = LstmPolicy
        elif policy == 'lnlstm':
            policy_fn = LnLstmPolicy
        self.learn(policy_fn, env, seed, total_timesteps=num_frames, lrschedule=lrschedule, nsteps=(1 if FLAGS.render else 5))
        env.close()

    def learn(self, policy, env, seed, nsteps=5, nstack=4, total_timesteps=int(80e6), vf_coef=0.5, ent_coef=0.01,
              max_grad_norm=0.5, lr=7e-4, lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100):
        tf.reset_default_graph()
        set_global_seeds(seed)

        nenvs = env.num_envs
        ob_space = env.observation_space
        ac_space = env.action_space
        num_procs = len(env.remotes)  # HACK
        model = Model(policy=policy, ob_space=ob_space, ac_space=ac_space, nenvs=nenvs, nsteps=nsteps, nstack=nstack,
                      num_procs=num_procs, ent_coef=ent_coef, vf_coef=vf_coef,
                      max_grad_norm=max_grad_norm, lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                      lrschedule=lrschedule)

        if FLAGS.load_model:
            model.load(FLAGS.load_model)

        runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)

        nbatch = nenvs * nsteps
        tstart = time.time()
        for update in range(1, total_timesteps // nbatch + 1):
            if FLAGS.action_args:
                obs, states, rewards, masks, actions, actionxs, actionys, values = runner.run()
            else:
                obs, states, rewards, masks, actions, values = runner.run()

            if not FLAGS.validate:
                if FLAGS.action_args:
                    policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, actionxs, actionys, values)
                else:
                    policy_loss, value_loss, policy_entropy = model.train(obs, states, rewards, masks, actions, values)
            else:
                policy_loss, value_loss, policy_entropy = [0, 0, 0]
            if FLAGS.render:
                time.sleep(0.33)
            nseconds = time.time() - tstart
            fps = int((update * nbatch) / nseconds)

            if update % 5000 == 0 and not FLAGS.validate:
                savepath = osp.join(logger.get_dir(), FLAGS.save_model, "checkpoint." + str(update))
                print('Saving to', savepath)
                model.save(savepath)

            if update % log_interval == 0 or update == 1:
                ev = explained_variance(values, rewards)
                logger.record_tabular("nupdates", update)
                logger.record_tabular("total_timesteps", update * nbatch)
                logger.record_tabular("fps", fps)
                logger.record_tabular("policy_entropy", float(policy_entropy))
                logger.record_tabular("value_loss", float(value_loss))
                logger.record_tabular("explained_variance", float(ev))
                logger.dump_tabular()
        savepath = osp.join(logger.get_dir(), FLAGS.save_model)
        print('Saving to', savepath)
        model.save(savepath)
        env.close()