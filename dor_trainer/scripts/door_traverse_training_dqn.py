import sys
import os
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from datetime import datetime
import logging
logging.basicConfig(format='%(asctime)s %(message)s',level=logging.INFO)

from agents.dqn_conv import DQNAgent
from envs.door_open_specific_envs import DoorTraverseTaskEnv, ModelSaver
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
    parser.add_argument('--pull_model', type=str ,default="dqn_noise0.0")
    return parser.parse_args()

if __name__=='__main__':
    args = get_args()
    rospy.init_node('dqn_train', anonymous=True, log_level=rospy.INFO)
    # parameter
    num_episodes = args.max_ep
    num_steps = args.max_step
    # instantiate env
    env = DoorTraverseTaskEnv(resolution=(64,64),cam_noise=args.noise, pull_policy='dqn', pull_model=args.pull_model)
    act_dim = env.action_dimension()

    train_freq = 80
    # variables
    step_counter = 0
    success_counter = 0
    episodic_returns = []
    sedimentary_returns = []
    ep_rew = 0
    # instantiate agent
    agent_p = DQNAgent(name='door_traverse',dim_img=(64,64,3),dim_act=act_dim)
    model_path = os.path.join(sys.path[0], 'saved_models', agent_p.name, 'models')

    # use tensorboard
    train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
    summary_writer = tf.summary.create_file_writer(model_path)
    summary_writer.set_as_default()

    model_saver = ModelSaver(500)

    start_time = time.time()
    for ep in range(num_episodes):
        obs, info = env.reset()
        ep_rew = 0
        img = obs.copy()
        agent_p.linear_epsilon_decay(curr_ep=ep)
        for st in range(num_steps):
            act = agent_p.epsilon_greedy(img)
            obs, rew, done, info = env.step(act)
            nxt_img = obs.copy()
            ep_rew += rew
            step_counter += 1
            # store transition
            agent_p.replay_buffer.store(img, act, rew, done, nxt_img)
            # train one step
            if ep >= agent_p.warmup_episodes:
                if not step_counter%train_freq:
                    for _ in range(train_freq):
                        agent_p.train_one_step(train_loss)
            # finish step, EXTREMELY IMPORTANT!!!
            img = nxt_img.copy()
            logging.debug("\n-\nepisode: {}, step: {} \naction: {} \nreward: {} \ndone: {}".format(ep+1, st+1, act, rew, done))
            if done:
                break

        # log infomation for each episode
        episodic_returns.append(ep_rew)
        sedimentary_returns.append(sum(episodic_returns)/(ep+1))
        if env.success:
            success_counter += 1

        tf.summary.scalar("episode total reward", ep_rew, step=ep)
        tf.summary.scalar('loss', train_loss.result(), step=ep)

        rospy.loginfo(
            "\n================\nEpisode: {} \nEpsilon: {} \nEpisodeLength: {} \nEpisodeTotalRewards: {} \nAveragedTotalReward: {} \nSuccess: {} \nTime: {} \n================\n".format(
                ep+1,
                agent_p.epsilon,
                st+1,
                ep_rew,
                sedimentary_returns[-1],
                success_counter,
                time.time()-start_time
            )
        )

        train_loss.reset_states()
        # save model
        model_saver.save(ep,model_path,agent_p.dqn_active)
