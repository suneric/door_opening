#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import numpy as np
import rospy
from .door_open_task_env import DoorOpenTaskEnv
from .door_open_specific_envs import DoorPullTaskEnv
import sys
import os
from math import *

# with Force Sensor
class MS_DoorPullTaskEnv(DoorPullTaskEnv):
    def __init__(self,resolution=(64,64),cam_noise=0.0):
        super(MS_DoorPullTaskEnv, self).__init__(resolution, cam_noise)
        self.safe_force = 270 #N

    def _set_init(self):
      self.driver.stop()
      self._reset_mobile_robot(1.5,0.5,0.075,3.14)
      self._wait_door_closed()
      self._random_init_mobile_robot()
      self.step_cnt = 0
      self.success = False
      self.force_sensor.reset_record()
      self.max_force = 0.0

    def _take_action(self, action_idx):
      _,self.door_angle = self._door_position()
      action = self.action_space[action_idx]
      self.driver.drive(action[0],action[1])
      rospy.sleep(0.5)
      self.max_force = self.force_sensor.data()
      self.step_cnt += 1

    def _is_done(self):
      self.success = self._door_is_open()
      done = False
      if self.max_force > self.safe_force:
          print("maximum forces", self.force_sensor.force_record())
          return True

      if self._door_pull_failed() or self._door_is_open():
          done = True
      return done

    def _compute_reward(self):
      reward = 0
      if self.max_force > self.safe_force:
        reward = -10
        return reward

      if self._door_is_open():
          reward = 100
      elif self._door_pull_failed():
          reward = -10
      else:
          door_r, door_a = self._door_position()
          delta_a = door_a-self.door_angle
          reward = delta_a*10 - 0.1
      return reward
