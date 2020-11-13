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

    def _set_init(self):
      self.driver.stop()
      self._reset_mobile_robot(1.5,0.5,0.075,3.14)
      self._wait_door_closed()
      self._random_init_mobile_robot()
      self.step_cnt = 0
      self.success = False
      self.force_sensor.reset_record()
      self.force_safe = True

    def _action_space(self):
      lv,av = 1.5,3.14
      safe_base = np.array([[lv,av],[lv,0],[0,av],[-lv,av],[-lv,0],[lv,-av],[0,-av],[-lv,-av]])
      low = 0.2*safe_base
      middle = safe_base
      high = 2*safe_base
      action_space = np.concatenate((low,middle,high),axis=0)
      print("action space", action_space)
      return action_space

    def _is_done(self):
      self.success = self._door_is_open()
      done = False
      if not self.force_safe:
          print("maximum forces", self.force_sensor.force_record())
          return True

      if self._door_pull_failed() or self._door_is_open():
          done = True
      return done

    def _compute_reward(self):
      reward = 0
      if not self.force_sensor.safe():
        reward = -10
        self.force_safe = False
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
