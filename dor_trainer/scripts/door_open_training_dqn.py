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
from envs.door_open_task_env import DoorOpenTaskEnv
import rospy
import tensorflow as tf

if __name__=='__main__':
    rospy.init_node('dqn_train', anonymous=True, log_level=rospy.INFO)
    # instantiate env
    env = DoorOpenTaskEnv(resolution=(64,64))
    # parameter
    num_episodes = 6000
    num_steps = env.max_episode_steps
    train_freq = 80
    # variables
    step_counter = 0
    success_counter = 0
    episodic_returns = []
    sedimentary_returns = []
    ep_rew = 0
    # instantiate agent
    agent_p = DQNAgent(name='door_open',dim_img=(64,64,3),dim_act=5)
    model_path = os.path.join(sys.path[0], 'saved_models', agent_p.name, 'models')

    # use tensorboard
    summary_writer = tf.summary.create_file_writer(model_path)
    summary_writer.set_as_default()

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
                        agent_p.train_one_step()
            # finish step, EXTREMELY IMPORTANT!!!
            img = nxt_img.copy()
            logging.debug("\n-\nepisode: {}, step: {} \naction: {} \nreward: {} \ndone: {}".format(ep+1, st+1, act, rew, done))
            if done:
                break

        # log infomation for each episode
        episodic_returns.append(ep_rew)
        sedimentary_returns.append(sum(episodic_returns)/(ep+1))
        if env.open:
            success_counter += 1

        tf.summary.scalar("episode total reward", ep_rew, step=ep)

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


    # plot averaged returns
    # fig, ax = plt.subplots(figsize=(8, 6))
    # fig.suptitle('Averaged Returns')
    # ax.plot(sedimentary_returns)
    # plt.show()

    # save model
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(model_path))
    agent_p.dqn_active.save(model_path)
