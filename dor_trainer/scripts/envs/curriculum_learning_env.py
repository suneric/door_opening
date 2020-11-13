#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import numpy as np
import rospy
from .door_open_task_env import DoorOpenTaskEnv
import sys
import os
from math import *
from tf.transformations import quaternion_from_euler, euler_from_matrix

class DoorPullAndTraverseTaskEnv(DoorOpenTaskEnv):
    def __init__(self,resolution=(64,64), cam_noise=0.0):
        super(DoorPullAndTraverseTaskEnv, self).__init__(resolution,cam_noise)
        self.door_angle = 0.1
        self.robot_x = 0.61

    def _set_init(self):
      self.driver.stop()
      self._reset_mobile_robot(1.5,0.5,0.075,3.14)
      self._wait_door_closed()
      self._random_init_mobile_robot()
      self.step_cnt = 0
      self.stage = 'pull'
      self.success = False

    def _random_init_mobile_robot(self):
        cx = 0.01*(np.random.uniform()-0.5)+0.07
        cy = 0.01*(np.random.uniform()-0.5)+0.95
        theta = 0.1*(np.random.uniform()-0.5)+pi
        camera_pose = np.array([[cos(theta),sin(theta),0,cx],
                                [-sin(theta),cos(theta),0,cy],
                                [0,0,1,0.075],
                                [0,0,0,1]])
        mat = np.array([[1,0,0,0.5],
                        [0,1,0,-0.25],
                        [0,0,1,0],
                        [0,0,0,1]])
        R = np.dot(camera_pose,np.linalg.inv(mat));
        euler = euler_from_matrix(R[0:3,0:3], 'rxyz')
        robot_x = R[0,3]
        robot_y = R[1,3]
        robot_z = R[2,3]
        yaw = euler[2]
        self._reset_mobile_robot(robot_x,robot_y,robot_z,yaw)

    def _take_action(self, action_idx):
      _,self.door_angle = self._door_position()
      self.robot_x = self.pose_sensor.robot().position.x
      action = self.action_space[action_idx]
      self.driver.drive(action[0],action[1])
      rospy.sleep(0.5)
      self.step_cnt += 1

    def _is_done(self):
        done = False
        if self.stage == 'pull':
            if self._door_pull_failed():
                done = True
            elif self._door_is_open(): # change stage
                self.stage = 'traverse'
        else:
            self.success = self._robot_is_out()
            if self._door_traverse_failed() or self._robot_is_out():
                done = True

        return done

    def _compute_reward(self):
        # divid to two stages, pull and traverse with different rewarding function
        reward = 0
        if self.stage == 'pull':
            if self._door_is_open():
                reward = 100
            elif self._door_pull_failed():
                reward = -10
            else:
                door_r, door_a = self._door_position()
                delta_a = door_a-self.door_angle
                reward = delta_a*10 - 0.1
        elif self.stage == 'traverse':
            if self._robot_is_out():
                reward = 100
            elif self._door_traverse_failed():
                reward = -10
            else:
                delta_x = -(self.pose_sensor.robot().position.x - self.robot_x)
                reward = delta_x*10 - 0.1

        return reward

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

    def _door_traverse_failed(self):
        if not self._robot_is_out():
            campose_r, campose_a = self._camera_position()
            doorpose_r, doorpos_a = self._door_position()
            if campose_r > 1.1*doorpose_r or campose_a > 1.1*doorpos_a:
                return True
        return False
#
