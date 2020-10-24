#!/usr/bin/env python
import sys
import os
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

from agents.ppo_conv import PPOAgent, PPOBuffer
from envs.door_open_specific_envs import DoorPushTaskEnv, ModelSaver
import rospy
import tensorflow as tf
import argparse

# application wise random seed
np.random.seed(7)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--max_ep', type=int, default=10000)
    parser.add_argument('--max_step', type=int, default=60)
    return parser.parse_args()

if __name__=='__main__':
    args = get_args()
    rospy.init_node('ppo_train', anonymous=True, log_level=rospy.INFO)
    env = DoorPushTaskEnv(resolution=(64,64), cam_noise=args.noise)
    dim_obs = (64,64,3)
    dim_act = env.action_dimension()
    agent = PPOAgent(
        env_type='discrete',
        dim_obs=dim_obs,
        dim_act=dim_act,
    )
    replay_buffer = PPOBuffer(dim_obs=dim_obs, dim_act=1, size=1000, gamma=0.99, lam=0.97)
    model_dir = os.path.join(sys.path[0], 'saved_models', 'door_open', agent.name+'_noise'+str(args.noise), datetime.now().strftime("%Y-%m-%d-%H-%M"))
    # paramas
    steps_per_epoch = replay_buffer.max_size
    # epochs = 100
    iter_a = 80
    iter_c = 80
    max_ep_len=args.max_step
    save_freq=10
    # get ready
    summary_writer = tf.summary.create_file_writer(model_dir)
    summary_writer.set_as_default()
    obs, info = env.reset()
    ep, ep_ret, ep_len = 0, 0, 0
    episode_counter, step_counter, success_counter = 0, 0, 0
    stepwise_rewards, episodic_returns, sedimentary_returns = [], [], []
    episodic_steps = []
    start_time = time.time()
    # main loop
    while episode_counter < args.max_ep:
        for st in range(steps_per_epoch):
            act, val, logp = agent.pi_of_a_given_s(np.expand_dims(obs, axis=0))
            n_obs, rew, done, info = env.step(act)
            rospy.logdebug(
                "\nepisode: {}, step: {} \nstate: {} \naction: {} \nreward: {} \ndone: {} \nn_state: {}".format(episode_counter+1, st+1, obs, act, rew, done, n_obs)
            )
            ep_ret += rew
            ep_len += 1
            stepwise_rewards.append(rew)
            step_counter += 1
            replay_buffer.store(obs, act, rew, val, logp)
            obs = n_obs # SUPER CRITICAL!!!
            # handle episode termination
            timeout = (ep_len==max_ep_len)
            terminal = done or timeout
            epoch_ended = (st==steps_per_epoch-1)
            if terminal or epoch_ended:
                episode_counter += 1
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at {} steps.'.format(ep_len))
                if timeout or epoch_ended:
                    _, val, _ = agent.pi_of_a_given_s(np.expand_dims(obs, axis=0))
                else:
                    val = 0
                replay_buffer.finish_path(val)
                if terminal:
                    episodic_returns.append(ep_ret)
                    sedimentary_returns.append(sum(episodic_returns)/episode_counter)
                    episodic_steps.append(step_counter)
                    if env.success:
                        success_counter += 1
                    rospy.loginfo(
                        "\n----\nTotalSteps: {} Episode: {}, EpReturn: {}, EpLength: {}, Succeeded: {}\n----\n".format(step_counter, episode_counter, ep_ret, ep_len, success_counter)
                    )
                tf.summary.scalar("episode total reward", ep_ret, step=episode_counter)
                obs, info = env.reset()
                ep_ret, ep_len = 0, 0
        # update actor-critic
        loss_pi, loss_v, loss_info = agent.train(replay_buffer.get(), iter_a, iter_c)
        rospy.loginfo("\n====\nEpoch: {} \nEpisodes: {} \nSteps: {} \nAveReturn: {} \nSucceeded: {} \nLossPi: {} \nLossV: {} \nKLDivergence: {} \nEntropy: {} \nTimeElapsed: {}\n====\n".format(ep+1, episode_counter, step_counter, sedimentary_returns[-1], success_counter, loss_pi, loss_v, loss_info['kl'], loss_info['ent'], time.time()-start_time))
        tf.summary.scalar('loss_pi', loss_pi, step=ep)
        tf.summary.scalar('loss_v', loss_v, step=ep)
        ep += 1
        # Save model
        if not ep%save_freq or (episode_counter >= args.max_ep):
            # save logits_net
            logits_net_path = os.path.join(model_dir, 'logits_net', str(ep))
            if not os.path.exists(os.path.dirname(logits_net_path)):
                os.makedirs(os.path.dirname(logits_net_path))
            agent.actor.logits_net.save(logits_net_path)
            # save val_net
            val_net_path = os.path.join(model_dir, 'val_net', str(ep))
            if not os.path.exists(os.path.dirname(val_net_path)):
                os.makedirs(os.path.dirname(val_net_path))
            agent.critic.val_net.save(val_net_path)
            # Save returns
            np.save(os.path.join(model_dir, 'episodic_returns.npy'), episodic_returns)
            np.save(os.path.join(model_dir, 'sedimentary_returns.npy'), sedimentary_returns)
            np.save(os.path.join(model_dir, 'episodic_steps.npy'), episodic_steps)
            with open(os.path.join(model_dir, 'training_time.txt'), 'w') as f:
                f.write("{}".format(time.time()-start_time))
