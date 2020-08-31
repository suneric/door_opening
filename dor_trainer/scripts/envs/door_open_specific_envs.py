#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import numpy as np
import rospy
from .gym_gazebo_env import GymGazeboEnv
from gym.envs.registration import register
from std_msgs.msg import Float64
from gazebo_msgs.msg import LinkStates, ModelStates, ModelState, LinkState
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import Image
from tf.transformations import quaternion_from_euler
import cv2
from cv_bridge import CvBridge, CvBridgeError
import math

#
class DoorPullTaskEnv(DoorOpenTaskEnv):
    def __init__(self,resolution=(64,64),noise=0.0):
        super(DoorPullTaskEnv, self).__init__(resolution,noise)
        self.door_angle = 0.1

    def _set_init(self):
      self.driver.stop()
      self._reset_mobile_robot(1.5,0.5,0.075,3.14)
      self._wait_door_closed()
      self._reset_mobile_robot(0.61,0.77,0.075,3.3)
      self.step_cnt = 0
      self.success = False

    def _take_action(self, action_idx):
      _,self.door_angle = self._door_position()
      action = self.action_space[action_idx]
      self.driver.drive(action[0],action[1])
      rospy.sleep(0.5)
      self.step_cnt += 1
      self.success = self._door_is_open()

    def _is_done(self):
      if self._door_pull_failed() or self._door_is_open():
          return True
      else:
          return False

    def _compute_reward(self):
      reward = 0
      if self.success:
          reward = 100
      else:
          door_r, door_a = self._door_position()
          delta_a = door_a-self.door_angle
          reward = delta_a*10

      # try to achieve less steps
      penalty = 0.1
      #print("step reward and penalty: ", reward, penalty)
      return reward-penalty

    # check the position of camera
    # if it is in the door block, still trying
    # else failed, reset env
    def _door_pull_failed(self):
        if not self._robot_is_out():
            campose_r, campose_a = self._camera_position()
            doorpose_r, doorpos_a = self._door_position()
            if campose_r > 1.1*doorpose_r or campose_a > 1.1*doorpos_a:
                return True
        return False
#
#
#
class DoorPushTaskEnv(DoorOpenTaskEnv):
    def __init__(self,resolution=(64,64),noise=0.0):
        super(DoorPushTaskEnv, self).__init__(resolution,noise)

    def _set_init(self):
      self.driver.stop()
      self._reset_mobile_robot(-1.0,0.5,0.075,0)
      self._wait_door_closed()
      self.step_cnt = 0
      self.success = False

    def _take_action(self, action_idx):
      action = self.action_space[action_idx]
      self.driver.drive(action[0],action[1])
      rospy.sleep(0.5)
      self.step_cnt += 1
      self.success = self._robot_is_in()

    def _is_done(self):
      if self._door_push_failed() or self._robot_is_in():
          return True
      else:
          return False

    def _compute_reward(self):
      reward = 0
      if self.success:
          reward = 100
      else:
          reward = self.pose_sensor.robot().position.x

      # try to achieve less steps
      # penalty = 0.1
      #print("step reward and penalty: ", reward, penalty)
      return reward

    def _door_push_failed(self):
        if self._robot_is_out():
            cam_pose = self._robot_footprint_position(0.5,-0.25)
            cam_x, cam_y = cam_pose[0,3], cam_pose[1,3]
            if cam_x < -1.0 or cam_y < -0.1 or cam_y > 1.1:
                return True
        return False

#
#
#
class DoorTraverseTaskEnv(DoorOpenTaskEnv):
    def __init__(self,resolution=(64,64),noise=0.0):
        super(DoorTraverseTaskEnv, self).__init__(resolution,noise)

    def _set_init(self):
      self.driver.stop()
      self._reset_mobile_robot(1.5,0.5,0.075,3.14)
      self._wait_door_closed()
      self._reset_mobile_robot(0.61,0.77,0.075,3.3)
      self.step_cnt = 0
      self._pull_door('door_open_3')
      self.success = False

    def _take_action(self, action_idx):
      action = self.action_space[action_idx]
      self.driver.drive(action[0],action[1])
      rospy.sleep(0.5)
      self.step_cnt += 1
      self.success = self._robot_is_out()

    def _is_done(self):
      if self._door_open_failed() or self._robot_is_out():
          return True
      else:
          return False

    def _compute_reward(self):
      reward = 0
      if self.success:
          reward = 100
      else:
          reward = -self.pose_sensor.robot().position.x

      # try to achieve less steps
      # penalty = 0.1
      #print("step reward and penalty: ", reward, penalty)
      return reward

    def _pull_door(self, dqn_model):
        agent = DQNAgent(name='door_pull',dim_img=(64,64,3),dim_act=act_dim)
        model_path = os.path.join(sys.path[0], 'trained_models', dqn_model, 'models')
        agent.dqn_active = tf.keras.models.load_model(model_path)
        agent.epsilon = 0.0 # determinitic action without random choice
        obs = self._get_observation()
        img = obs.copy()
        step_cnt = 0
        while not self._door_is_open():
            act_idx = agent.epsilon_greedy(img)
            self.gazebo.unpauseSim()
            action = self.action_space[act_idx]
            self.driver.drive(action[0],action[1])
            rospy.sleep(0.5)
            self.gazebo.pauseSim()
            obs = self._get_observation()
            img = obs.copy()
            step_cnt += 1
            if step_cnt > self.max_episode_steps:
                print("door pull exceeds max steps")
                break
