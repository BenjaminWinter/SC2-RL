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
import numpy as np

FLAGS = flags.FLAGS

flags.DEFINE_integer('seed', 0, 'RNG Seed')

if FLAGS.action_args:
    from baselines_mod.acktr.acktr_disc import Runner, Model
    from baselines_mod.acktr.policies_coords import CnnPolicy
else:
    from baselines.acktr.acktr_disc import Runner, Model
    from baselines_mod.acktr.policies import CnnPolicy

class Acktr:
    def __init__(self, run_time, a_space, s_space):
        pass

    def run(self):
        self.train(num_frames=FLAGS.run_time, seed=FLAGS.seed, num_cpu=FLAGS.threads)

    def train(self, num_frames, seed, num_cpu):

        def make_env(rank):
            def _thunk():
                env = helpers.get_env_wrapper(render=FLAGS.render)
                env.seed(seed + rank)
                if logger.get_dir():
                    env = bench.Monitor(env, os.path.join(logger.get_dir(), "{}.monitor.json".format(rank)))
                gym.logger.setLevel(logging.WARN)
                return env

            return _thunk

        set_global_seeds(seed)
        if FLAGS.validate:
            from util.environments.sim_proc_env import SimProcEnv
            env = SimProcEnv(render=True)
        else:
            env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
        policy_fn = CnnPolicy
        self.learn_disc(policy_fn, env, seed, total_timesteps=num_frames, nprocs=num_cpu, nsteps=(1 if FLAGS.render else 20), lr=FLAGS.lr)
        env.close()

    def learn_disc(self, policy, env, seed, total_timesteps=int(40e6), gamma=0.99, log_interval=1, nprocs=32, nsteps=20,
              nstack=4, ent_coef=0.01, vf_coef=0.5, vf_fisher_coef=1.0, lr=0.25, max_grad_norm=0.5,
              kfac_clip=0.001, save_interval=None, lrschedule='linear'):
        tf.reset_default_graph()
        set_global_seeds(seed)

        nenvs = env.num_envs

        ob_space = env.observation_space
        ac_space = env.action_space
        make_model = lambda: Model(policy, ob_space, ac_space, nenvs, total_timesteps, nprocs=nprocs, nsteps
        =nsteps, nstack=nstack, ent_coef=ent_coef, vf_coef=vf_coef, vf_fisher_coef=
                                   vf_fisher_coef, lr=lr, max_grad_norm=max_grad_norm, kfac_clip=kfac_clip,
                                   lrschedule=lrschedule)
        if save_interval and logger.get_dir():
            import cloudpickle
            with open(osp.join(logger.get_dir(), 'make_model.pkl'), 'wb') as fh:
                fh.write(cloudpickle.dumps(make_model))
        model = make_model()
        if FLAGS.load_model:
            model.load(FLAGS.load_model)

        runner = Runner(env, model, nsteps=nsteps, nstack=nstack, gamma=gamma)
        nbatch = nenvs * nsteps
        tstart = time.time()
        enqueue_threads = model.q_runner.create_threads(model.sess, coord=tf.train.Coordinator(), start=True)
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
                policy_loss, value_loss, policy_entropy = [0,0,0]
            if FLAGS.render:
                time.sleep(0.03)

            model.old_obs = obs
            nseconds = time.time() - tstart
            fps = int((update * nbatch) / nseconds)
            avg_reward = np.average(rewards)

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
                logger.record_tabular("policy_loss", float(policy_loss))
                logger.record_tabular("value_loss", float(value_loss))
                logger.record_tabular("explained_variance", float(ev))
                logger.record_tabular("avg reward", float(avg_reward))
                logger.dump_tabular()

        savepath = osp.join(logger.get_dir(), FLAGS.save_model, "final")
        print('Saving to', savepath)
        model.save(savepath)

        env.close()

