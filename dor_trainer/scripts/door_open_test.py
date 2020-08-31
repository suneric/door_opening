#!/usr/bin/env python
from __future__ import print_function

import sys
import os
import gym
import numpy as np
import time
import random
from gym import wrappers
# ROS packages required
import rospy
import rospkg
from geometry_msgs.msg import Pose
from geometry_msgs.msg import Twist
from gazebo_msgs.msg import ModelStates, LinkStates
import tensorflow as tf
import pickle
import argparse

from agents.dqn_conv import DQNAgent
from envs.door_open_specific_envs import DoorPullTaskEnv, DoorPushTaskEnv, DoorTraverseTaskEnv

def random_test(episode):
    env = DoorOpenTaskEnv()
    act_dim = env.action_dimension()
    # test env with random sampled actions
    steps = env.max_episode_steps
    for ep in range(episode):
        state, info = env.reset()
        done = False
        for step in range(steps):
            action_idx = np.random.randint(act_dim)
            next_state, reward, done, info = env.step(action_idx)
            print("Episode : {}, Step: {}, \n current pose.x: {},, Reward: {:.4f}".format(
                episode,
                step,
                info.position.x,
                reward
            ))
            if done:
                break

def dqn_pull_test(episode):
    env = DoorPullTaskEnv(resolution=(64,64))
    act_dim = env.action_dimension()
    agent = DQNAgent(name='door_pull',dim_img=(64,64,3),dim_act=act_dim)
    model_path = os.path.join(sys.path[0], 'saved_models', agent.name, 'models')
    agent.dqn_active = tf.keras.models.load_model(model_path)

    steps = env.max_episode_steps
    start_time = time.time()
    success_counter = 0
    episodic_returns = []
    sedimentary_returns = []
    ep_rew = 0
    agent.epsilon = 0.0
    for ep in range(episode):
        obs, info = env.reset()
        ep_rew = 0
        img = obs.copy()
        for st in range(steps):
            act = agent.epsilon_greedy(img)
            obs, rew, done, info = env.step(act)
            img = obs.copy()
            ep_rew += rew
            if done:
                break

        # log infomation for each episode
        episodic_returns.append(ep_rew)
        sedimentary_returns.append(sum(episodic_returns)/(ep+1))
        if env.success:
            success_counter += 1

        rospy.loginfo(
            "\n================\nEpisode: {} \nEpisodeLength: {} \nEpisodeTotalRewards: {} \nAveragedTotalReward: {} \nSuccess: {} \nTime: {} \n================\n".format(
                ep+1,
                st+1,
                ep_rew,
                sedimentary_returns[-1],
                success_counter,
                time.time()-start_time
            )
        )

def dqn_push_test(episode):
    env = DoorPushTaskEnv(resolution=(64,64))
    act_dim = env.action_dimension()
    agent = DQNAgent(name='door_push',dim_img=(64,64,3),dim_act=act_dim)
    model_path = os.path.join(sys.path[0], 'saved_models', agent.name, 'models')
    agent.dqn_active = tf.keras.models.load_model(model_path)

    steps = env.max_episode_steps
    start_time = time.time()
    success_counter = 0
    episodic_returns = []
    sedimentary_returns = []
    ep_rew = 0
    agent.epsilon = 0.0
    for ep in range(episode):
        obs, info = env.reset()
        ep_rew = 0
        img = obs.copy()
        for st in range(steps):
            act = agent.epsilon_greedy(img)
            obs, rew, done, info = env.step(act)
            img = obs.copy()
            ep_rew += rew
            if done:
                break

        # log infomation for each episode
        episodic_returns.append(ep_rew)
        sedimentary_returns.append(sum(episodic_returns)/(ep+1))
        if env.success:
            success_counter += 1

        rospy.loginfo(
            "\n================\nEpisode: {} \nEpisodeLength: {} \nEpisodeTotalRewards: {} \nAveragedTotalReward: {} \nSuccess: {} \nTime: {} \n================\n".format(
                ep+1,
                st+1,
                ep_rew,
                sedimentary_returns[-1],
                success_counter,
                time.time()-start_time
            )
        )

def dqn_traverse_test(episode):
    env = DoorTraverseTaskEnv(resolution=(64,64))
    act_dim = env.action_dimension()
    agent = DQNAgent(name='door_traverse',dim_img=(64,64,3),dim_act=act_dim)
    model_path = os.path.join(sys.path[0], 'saved_models', agent.name, 'models')
    agent.dqn_active = tf.keras.models.load_model(model_path)

    steps = env.max_episode_steps
    start_time = time.time()
    success_counter = 0
    episodic_returns = []
    sedimentary_returns = []
    ep_rew = 0
    agent.epsilon = 0.0
    for ep in range(episode):
        obs, info = env.reset()
        ep_rew = 0
        img = obs.copy()
        for st in range(steps):
            act = agent.epsilon_greedy(img)
            obs, rew, done, info = env.step(act)
            img = obs.copy()
            ep_rew += rew
            if done:
                break

        # log infomation for each episode
        episodic_returns.append(ep_rew)
        sedimentary_returns.append(sum(episodic_returns)/(ep+1))
        if env.success:
            success_counter += 1

        rospy.loginfo(
            "\n================\nEpisode: {} \nEpisodeLength: {} \nEpisodeTotalRewards: {} \nAveragedTotalReward: {} \nSuccess: {} \nTime: {} \n================\n".format(
                ep+1,
                st+1,
                ep_rew,
                sedimentary_returns[-1],
                success_counter,
                time.time()-start_time
            )
        )

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dqn', type=string, default="pull") # dgn test 1 or random test 0
    parser.add_argument('--eps', type=int, default=10) # test episode
    return parser.parse_args()


np.random.seed(7)

if __name__ == "__main__":
    args = get_args()
    rospy.init_node('env_test', anonymous=True, log_level=rospy.INFO)
    if args.dqn == "pull":
        dqn_pull_test(args.eps)
    elif args.dqn == "push":
        dqn_push_test(args.eps)
    elif args.dqn == "traverse":
        dqn_traverse_test(args.eps)
    else:
        random_test(args.eps)
