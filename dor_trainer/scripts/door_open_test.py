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

def dqn_pull_test(episode, path,model_name,act_dim):
    env = DoorPullTaskEnv(resolution=(64,64))
    # act_dim = env.action_dimension()
    agent = DQNAgent(name='door_pull',dim_img=(64,64,3),dim_act=act_dim)
    model_path = os.path.join(sys.path[0], path, model_name, 'models')
    agent.dqn_active = tf.keras.models.load_model(model_path)

    steps = env.max_episode_steps
    start_time = time.time()
    success_counter = 0
    episodic_returns = []
    sedimentary_returns = []
    ep_rew = 0
    agent.epsilon = 0.0
    success_steps = []
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

        if env.success:
            success_counter += 1
            success_steps.append(st)
    return success_counter, np.mean(success_steps)


def dqn_push_test(episode,path, model_name):
    env = DoorPushTaskEnv(resolution=(64,64))
    act_dim = env.action_dimension()
    agent = DQNAgent(name='door_push',dim_img=(64,64,3),dim_act=act_dim)
    model_path = os.path.join(sys.path[0], path, model_name, 'models')
    agent.dqn_active = tf.keras.models.load_model(model_path)

    steps = env.max_episode_steps
    start_time = time.time()
    success_counter = 0
    episodic_returns = []
    sedimentary_returns = []
    ep_rew = 0
    agent.epsilon = 0.0
    success_steps = []
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
        if env.success:
            success_counter += 1
            success_steps.append(st)

    return success_counter, np.mean(success_steps)

def dqn_traverse_test(episode,path, model_name):
    env = DoorTraverseTaskEnv(resolution=(64,64))
    act_dim = env.action_dimension()
    agent = DQNAgent(name='door_traverse',dim_img=(64,64,3),dim_act=act_dim)
    model_path = os.path.join(sys.path[0], path, model_name, 'models')
    agent.dqn_active = tf.keras.models.load_model(model_path)

    steps = env.max_episode_steps
    start_time = time.time()
    success_counter = 0
    episodic_returns = []
    sedimentary_returns = []
    ep_rew = 0
    agent.epsilon = 0.0
    success_steps = []
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
        if env.success:
            success_counter += 1
            success_steps.append(st)
    return success_counter, np.mean(success_steps)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dqn', type=str, default="pull") # dgn test 1 or random test 0
    parser.add_argument('--model', type=str, default="door_pull") # baseline
    parser.add_argument('--eps', type=int, default=10) # test episode
    parser.add_argument('--dim',type=int, default=8)
    parser.add_argument('--path',type=str, default="saved_models")
    return parser.parse_args()


np.random.seed(7)

if __name__ == "__main__":
    args = get_args()
    rospy.init_node('env_test', anonymous=True, log_level=rospy.INFO)
    if args.dqn == "pull":
        success_count, average_steps = dqn_pull_test(args.eps, args.path, args.model,args.dim)
        print("success", success_count,"/", args.eps)
        print("average steps", average_steps)
    elif args.dqn == "push":
        success_count, average_steps = dqn_push_test(args.eps, args.path, args.model)
        print("success", success_count,"/", args.eps)
        print("average steps", average_steps)
    elif args.dqn == "traverse":
        success_count, average_steps = dqn_traverse_test(args.eps, args.path, args.model)
        print("success", success_count,"/", args.eps)
        print("average steps", average_steps)
    else:
        print("choose task type, model name and number of test episode.")
