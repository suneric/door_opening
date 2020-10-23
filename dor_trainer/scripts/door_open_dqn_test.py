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
import matplotlib
import matplotlib.pyplot as plt
import math

from agents.dqn_conv import DQNAgent
from envs.door_open_specific_envs import DoorPullTaskEnv, DoorPushTaskEnv, DoorTraverseTaskEnv

# plot trajectory which contains a sequence of pose of robot and door
# info {
#       'door': (radius, angle),
#       'robot': [(fp_lf_x, fp_lf_y),
#                 (fp_rf_x, fp_rf_y),
#                 (fp_lr_x, fp_lr_y),
#                 (fp_rr_x, fp_rr_y),
#                 (cam_p_x, cam_p_y)]
#      }
def plot_trajectorty(t):
    # room
    plt.figure(1)
    font = {'weight':'bold','size':15}
    matplotlib.rc('font',**font)
    plt.xlim(-1.5,2.5)
    plt.ylim(-3,1)
    plt.axis('equal')
    rx = [0,0,3,3,0,0]
    ry = [0,-1,-1,2,2,0.95]
    plt.plot(rx,ry,linewidth=10.0,color='lightgrey')

    cost = round(trajectory_cost(t),3)
    txt_info = "Sidebar displacement: " + str(cost) + " m"
    plt.text(0,-0.5,txt_info,fontsize=15)

    #plt.grid(True)
    door_t = []
    robot_t = []
    cam_t = []
    index = 0
    for info in t:
        dr = info['door'][0]
        da = info['door'][1]
        door_t.append((dr*math.sin(da),dr*math.cos(da)))
        r1 = info['robot'][1]
        r2 = info['robot'][0]
        r3 = info['robot'][2]
        r4 = info['robot'][3]
        r5 = r1
        r6 = info['robot'][4]
        rx = np.mean([r1[0],r2[0],r3[0],r4[0]])
        ry = np.mean([r1[1],r2[1],r3[1],r4[1]])
        robot_t.append([rx,ry])
        cam_t.append(r6)
        # draw first and last footprint
        if index == len(t)-1:
            line1, = plt.plot([0,dr*math.sin(da)],[0,dr*math.cos(da)],linewidth=5,color='y')
            line1.set_label('door')
            line2, = plt.plot([r1[0],r2[0],r3[0],r4[0],r5[0]],[r1[1],r2[1],r3[1],r4[1],r5[1]],linewidth=3,color='dimgrey')
            line2.set_label('mobile robot')
            line3, = plt.plot([r5[0],r6[0]],[r5[1],r6[1]],linewidth=3,color='red')
            line3.set_label('sidebar')
            # plt.legend((line1,line2,line3),('door','mobile robot','sidebar'))
        elif index == 0 or index == int(len(t)/2) or index == int(len(t)/4) or index == int(len(t)*3/4):
            plt.plot([0,dr*math.sin(da)],[0,dr*math.cos(da)],linewidth=5,color='y',alpha=0.35)
            plt.plot([r1[0],r2[0],r3[0],r4[0],r5[0]],[r1[1],r2[1],r3[1],r4[1],r5[1]],linewidth=3,color='dimgrey',alpha=0.35)
            plt.plot([r5[0],r6[0]],[r5[1],r6[1]],linewidth=3,color='red',alpha=0.35)
        index += 1
    #plt.plot(np.matrix(door_t)[:,0],np.matrix(door_t)[:,1], linewidth=1.0, color='y')
    line4, = plt.plot(np.matrix(robot_t)[:,0],np.matrix(robot_t)[:,1], linewidth=1.0, color='dimgrey', linestyle='dashed')
    line4.set_label('trajectory of mobile robot')
    line5, = plt.plot(np.matrix(cam_t)[:,0],np.matrix(cam_t)[:,1], linewidth=1.0, color='red',linestyle='dashed')
    line5.set_label('trajectory of sidebar')
    plt.legend(loc='upper center')
    plt.show()

# a simple cost evaluation based on the end of the bar where camera placed at
# total displacement of the end of the bar
def trajectory_cost(t):
    dist = 0
    pos = t[0]['robot'][4]
    for info in t:
        cam = info['robot'][4]
        displacement = math.sqrt((cam[0]-pos[0])*(cam[0]-pos[0])+(cam[1]-pos[1])*(cam[1]-pos[1]))
        pos = cam
        dist += displacement
    return dist

def dqn_pull_test(episode, path,model_name,act_dim,noise):
    env = DoorPullTaskEnv(resolution=(64,64),cam_noise=noise)
    # act_dim = env.action_dimension()
    agent = DQNAgent(name='door_pull',dim_img=(64,64,3),dim_act=act_dim)
    model_path = os.path.join(sys.path[0], path, model_name, 'models')
    agent.dqn_active = tf.keras.models.load_model(model_path)

    steps = 60
    start_time = time.time()
    success_counter = 0
    episodic_returns = []
    sedimentary_returns = []
    ep_rew = 0
    agent.epsilon = 0.0
    success_steps = []
    trajectories = []
    for ep in range(episode):
        trajectory = []
        obs, info = env.reset()
        trajectory.append(info)
        ep_rew = 0
        img = obs.copy()
        for st in range(steps):
            act = agent.epsilon_greedy(img)
            obs, rew, done, info = env.step(act)
            trajectory.append(info)
            img = obs.copy()
            ep_rew += rew
            if done:
                break

        # log infomation for each episode
        episodic_returns.append(ep_rew)
        sedimentary_returns.append(sum(episodic_returns)/(ep+1))
        if env.success:
            success_counter += 1
            success_steps.append(st)
            trajectories.append(trajectory)
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

    # find the min length of trajectory in all trajectories
    shortest_trajectory = min(trajectories, key=len)
    trajectory_costs = [round(trajectory_cost(i),3) for i in trajectories]
    return success_counter, np.mean(success_steps), shortest_trajectory, trajectory_costs


def dqn_push_test(episode,path,model_name,noise):
    env = DoorPushTaskEnv(resolution=(64,64),cam_noise=noise)
    act_dim = env.action_dimension()
    agent = DQNAgent(name='door_push',dim_img=(64,64,3),dim_act=act_dim)
    model_path = os.path.join(sys.path[0], path, model_name, 'models')
    agent.dqn_active = tf.keras.models.load_model(model_path)

    steps = 60
    start_time = time.time()
    success_counter = 0
    episodic_returns = []
    sedimentary_returns = []
    ep_rew = 0
    agent.epsilon = 0.0
    success_steps = []
    trajectories = []
    for ep in range(episode):
        trajectory = []
        obs, info = env.reset()
        trajectory.append(info)
        ep_rew = 0
        img = obs.copy()
        for st in range(steps):
            act = agent.epsilon_greedy(img)
            obs, rew, done, info = env.step(act)
            trajectory.append(info)
            img = obs.copy()
            ep_rew += rew
            if done:
                break

        # log infomation for each episode
        episodic_returns.append(ep_rew)
        sedimentary_returns.append(sum(episodic_returns)/(ep+1))
        if env.success:
            success_counter += 1
            success_steps.append(st)
            trajectories.append(trajectory)

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

    shortest_trajectory = min(trajectories, key=len)
    return success_counter, np.mean(success_steps), shortest_trajectory

def dqn_traverse_test(episode,path,model_name,noise):
    env = DoorTraverseTaskEnv(resolution=(64,64),cam_noise=noise)
    act_dim = env.action_dimension()
    agent = DQNAgent(name='door_traverse',dim_img=(64,64,3),dim_act=act_dim)
    model_path = os.path.join(sys.path[0], path, model_name, 'models')
    agent.dqn_active = tf.keras.models.load_model(model_path)

    steps = 60
    start_time = time.time()
    success_counter = 0
    episodic_returns = []
    sedimentary_returns = []
    ep_rew = 0
    agent.epsilon = 0.0
    success_steps = []
    trajectories = []
    for ep in range(episode):
        trajectory = []
        obs, info = env.reset()
        trajectory.append(info)
        ep_rew = 0
        img = obs.copy()
        for st in range(steps):
            act = agent.epsilon_greedy(img)
            obs, rew, done, info = env.step(act)
            trajectory.append(info)
            img = obs.copy()
            ep_rew += rew
            if done:
                break

        # log infomation for each episode
        episodic_returns.append(ep_rew)
        sedimentary_returns.append(sum(episodic_returns)/(ep+1))
        if env.success:
            success_counter += 1
            success_steps.append(st)
            trajectories.append(trajectory)

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

    shortest_trajectory = min(trajectories, key=len)
    return success_counter, np.mean(success_steps), shortest_trajectory

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dqn', type=str, default="pull") # dgn test 1 or random test 0
    parser.add_argument('--model', type=str, default="door_pull") # baseline
    parser.add_argument('--eps', type=int, default=10) # test episode
    parser.add_argument('--dim',type=int, default=8)
    parser.add_argument('--path',type=str, default="saved_models")
    parser.add_argument('--noise', type=float, default=0.0)
    return parser.parse_args()


np.random.seed(7)

if __name__ == "__main__":
    args = get_args()
    rospy.init_node('env_test', anonymous=True, log_level=rospy.INFO)
    if args.dqn == "pull":
        success_count, average_steps, shortest_trajectory, costs = dqn_pull_test(args.eps, args.path, args.model,args.dim, args.noise)
        print("success", success_count,"/", args.eps)
        print("average steps", average_steps)
        print("average cost", np.mean(costs), "minimum cost", np.min(costs), "maximum cost", np.max(costs))
        plot_trajectorty(shortest_trajectory)
    elif args.dqn == "push":
        success_count, average_steps, shortest_trajectory = dqn_push_test(args.eps, args.path, args.model, args.noise)
        print("success", success_count,"/", args.eps)
        print("average steps", average_steps)
        plot_trajectorty(shortest_trajectory)
    elif args.dqn == "traverse":
        success_count, average_steps, shortest_trajectory = dqn_traverse_test(args.eps, args.path, args.model, args.noise)
        print("success", success_count,"/", args.eps)
        print("average steps", average_steps)
        plot_trajectorty(shortest_trajectory)
    else:
        print("choose task type, model name and number of test episode.")
