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
from agents.ppo_conv import PPOAgent
from envs.door_open_specific_envs import DoorPullTaskEnv, DoorPushTaskEnv, DoorTraverseTaskEnv, DoorPullAndTraverseTaskEnv

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
    font = {'size':12}
    matplotlib.rc('font',**font)
    plt.xlim(-1.5,2.5)
    plt.ylim(-3,1)
    plt.axis('equal')
    rx = [0,0,3,3,0,0]
    ry = [0,-1,-1,2,2,0.95]
    plt.plot(rx,ry,linewidth=10.0,color='lightgrey')

    cost = round(trajectory_cost(t),3)
    txt_info = "Sidebar displacement: " + str(cost) + " m"
    plt.text(-0.1,-0.8,txt_info,fontsize=15)

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
    #plt.legend(loc='upper right')
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


###############################################################################
# DQN TEST

def run_dqn_test(episode,env,agent,max_steps):
    success_counter = 0
    trajectories = []
    values = []
    for ep in range(episode):
        trajectory = []
        vals = []
        obs, info = env.reset()
        trajectory.append(info)
        img = obs.copy()
        for st in range(max_steps):
            act = agent.epsilon_greedy(img)
            v = np.max(agent.dqn_active(np.expand_dims(img, axis=0)))
            vals.append(v)
            obs,rew,done,info = env.step(act)
            trajectory.append(info)
            img = obs.copy()
            if done:
                break

        if env.success:
            success_counter += 1
            trajectories.append(trajectory)
            values.append(vals)

        print("Succeeded: {} / {}".format(success_counter,ep+1))

    return trajectories, values

def dqn_pull_test(episode,model,noise):
    env = DoorPullTaskEnv(resolution=(64,64),cam_noise=noise)
    act_dim = env.action_dimension()
    agent = DQNAgent(name='door_pull',dim_img=(64,64,3),dim_act=act_dim)
    model_path = os.path.join(sys.path[0], "policy", "door_pull", model)
    agent.dqn_active = tf.keras.models.load_model(model_path)
    agent.epsilon = 0.0
    return run_dqn_test(episode,env,agent,60)

def dqn_push_test(episode,model,noise):
    env = DoorPushTaskEnv(resolution=(64,64),cam_noise=noise)
    act_dim = env.action_dimension()
    agent = DQNAgent(name='door_push',dim_img=(64,64,3),dim_act=act_dim)
    model_path = os.path.join(sys.path[0], "policy", "door_push", model)
    agent.dqn_active = tf.keras.models.load_model(model_path)
    agent.epsilon = 0.0
    return run_dqn_test(episode,env,agent,60)

def dqn_traverse_test(episode,model,noise):
    env = DoorTraverseTaskEnv(resolution=(64,64),cam_noise=noise,pull_policy='dqn',pull_model=model)
    act_dim = env.action_dimension()
    agent = DQNAgent(name='door_traverse',dim_img=(64,64,3),dim_act=act_dim)
    model_path = os.path.join(sys.path[0], "policy", "door_traverse", model)
    agent.dqn_active = tf.keras.models.load_model(model_path)
    agent.epsilon = 0.0
    return run_dqn_test(episode,env,agent,60)

def dqn_pull_traverse_test(episode,model,noise):
    env = DoorPullAndTraverseTaskEnv(resolution=(64,64),cam_noise=noise)
    act_dim = env.action_dimension()
    agent = DQNAgent(name='door_pull',dim_img=(64,64,3),dim_act=act_dim)
    model_path = os.path.join(sys.path[0], "policy", "door_open_traverse", model)
    agent.dqn_active = tf.keras.models.load_model(model_path)
    agent.epsilon = 0.0
    return run_dqn_test(episode,env,agent,100)

###############################################################################
# PPO TEST
def run_ppo_test(episode,env,agent,max_steps):
    success_counter = 0
    trajectories = []
    values = []
    for ep in range(episode):
        trajectory = []
        vals = []
        obs, info = env.reset()
        trajectory.append(info)
        done = False
        for st in range(max_steps):
            act, _, _ = agent.pi_of_a_given_s(np.expand_dims(obs, axis=0))
            v = np.squeeze(agent.critic.val_net(np.expand_dims(obs, axis=0)))
            vals.append(v)
            n_obs, rew, done, info = env.step(act)
            trajectory.append(info)
            obs = n_obs.copy()
            if done:
                break

        if env.success:
            success_counter += 1
            trajectories.append(trajectory)
            values.append(vals)

        print("Succeeded: {} / {}".format(success_counter,ep+1))

    return trajectories, values

def ppo_pull_test(episode,actor_model,critic_model,noise):
    env = DoorPullTaskEnv(resolution=(64,64),cam_noise=noise)
    act_dim = env.action_dimension()
    agent = PPOAgent(env_type='discrete',dim_obs=(64,64,3),dim_act=act_dim)
    actor_path = os.path.join(sys.path[0], "policy", "door_pull", actor_model)
    critic_path = os.path.join(sys.path[0], "policy", "door_pull", critic_model)
    agent.actor.logits_net = tf.keras.models.load_model(actor_path)
    agent.critic.val_net = tf.keras.models.load_model(critic_path)
    return run_ppo_test(episode,env,agent,60)

def ppo_push_test(episode,actor_model,critic_model,noise):
    env = DoorPushTaskEnv(resolution=(64,64),cam_noise=noise)
    act_dim = env.action_dimension()
    agent = PPOAgent(env_type='discrete',dim_obs=(64,64,3),dim_act=act_dim)
    actor_path = os.path.join(sys.path[0], "policy", "door_push", actor_model)
    critic_path = os.path.join(sys.path[0], "policy", "door_push", critic_model)
    agent.actor.logits_net = tf.keras.models.load_model(actor_path)
    agent.critic.val_net = tf.keras.models.load_model(critic_path)
    return run_ppo_test(episode,env,agent,60)

def ppo_traverse_test(episode,actor_model,critic_model,noise,model):
    env = DoorTraverseTaskEnv(resolution=(64,64),cam_noise=noise,pull_policy='ppo',pull_model=model)
    act_dim = env.action_dimension()
    agent = PPOAgent(env_type='discrete',dim_obs=(64,64,3),dim_act=act_dim)
    actor_path = os.path.join(sys.path[0], "policy", "door_traverse", actor_model)
    critic_path = os.path.join(sys.path[0], "policy", "door_traverse", critic_model)
    agent.actor.logits_net = tf.keras.models.load_model(actor_path)
    agent.critic.val_net = tf.keras.models.load_model(critic_path)
    return run_ppo_test(episode,env,agent,60)

def ppo_pull_traverse_test(episode, actor_model, critic_model, noise):
    env = DoorPullAndTraverseTaskEnv(resolution=(64,64),cam_noise=noise)
    act_dim = env.action_dimension()
    agent = PPOAgent(env_type='discrete',dim_obs=(64,64,3),dim_act=act_dim)
    actor_path = os.path.join(sys.path[0], "policy", "door_pull_traverse", actor_model)
    critic_path = os.path.join(sys.path[0], "policy", "door_pull_traverse", critic_model)
    agent.actor.logits_net = tf.keras.models.load_model(actor_path)
    agent.critic.val_net = tf.keras.models.load_model(critic_path)
    return run_ppo_test(episode,env,agent,100)


###############################################################################
# main loop
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default="pull") # pull, push, traverse
    parser.add_argument('--policy', type=str, default="dqn") # dqn, ppo
    parser.add_argument('--eps', type=int, default=10) # test episode
    parser.add_argument('--model',type=str, default="dqn_noise0.0") # model folder in folder "policy"
    parser.add_argument('--noise', type=float, default=0.0) # noise variance 0.02
    parser.add_argument('--actor_model',type=str, default="ppo_noise0.0/logits_net/150") # ppo model in folder "policy"
    parser.add_argument('--critic_model',type=str, default="ppo_noise0.0/val_net/150") # ppo model in folder "policy"
    return parser.parse_args()

np.random.seed(7)

if __name__ == "__main__":
    args = get_args()
    rospy.init_node('door_open_test', anonymous=True, log_level=rospy.INFO)

    # run specific task with specific policy
    trajectories = []
    if args.policy == "dqn":
        if args.task == "pull":
            trajectories, values = dqn_pull_test(args.eps,args.model,args.noise)
        elif args.task == "push":
            trajectories, values = dqn_push_test(args.eps,args.model,args.noise)
        elif args.task == "traverse":
            trajectories, values = dqn_traverse_test(args.eps,args.model,args.noise)
        elif args.task == "pull_traverse":
            trajectories, values = dqn_pull_traverse_test(args.eps,args.model,args.noise)

    elif args.policy == "ppo":
        if args.task == "pull":
            trajectories, values = ppo_pull_test(args.eps,args.actor_model,args.critic_model,args.noise)
        elif args.task == "push":
            trajectories, values = ppo_push_test(args.eps,args.actor_model,args.critic_model,args.noise)
        elif args.task == "traverse":
            trajectories, values = ppo_traverse_test(args.eps,args.actor_model,args.critic_model,args.noise,args.model)
        elif args.task == "pull_traverse":
            trajectories, values = ppo_pull_traverse_test(args.eps,args.actor_model,args.critic_model,args.noise)

    if len(trajectories) == 0:
        print("no successful test");
    else:
        # trajectory analysis
        trajectory_steps = [len(i)-1 for i in trajectories]
        average_steps = int(sum(trajectory_steps)/len(trajectory_steps))
        trajectory_costs = [round(trajectory_cost(i),3) for i in trajectories]
        average_values = [sum(vals)/len(vals) for vals in values]
        highest_values = [max(vals) for vals in values]
        lowest_values = [min(vals) for vals in values]
        print("====================")
        print("Success rate", len(trajectory_steps),"/", args.eps)
        print("Average steps", average_steps)
        print("Minimum steps", min(trajectory_steps))
        print("Maximum steps", max(trajectory_steps))
        print("Average Cost", sum(trajectory_costs)/len(trajectory_costs), "meters")
        print("Lowest Cost",  min(trajectory_costs), "meters")
        print("Highest Cost", max(trajectory_costs), "meters")
        print("Average Value", sum(average_values)/len(average_values))
        print("Lowest Value",  min(lowest_values))
        print("Highest Value", max(highest_values))
        print("====================")

        # plot trajectory with lowest cost
        lowest_index = trajectory_costs.index(min(trajectory_costs))
        plot_trajectorty(trajectories[lowest_index])
