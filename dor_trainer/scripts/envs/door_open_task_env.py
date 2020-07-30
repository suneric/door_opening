#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import numpy as np
import rospy
from gym import spaces
from .gym_gazebo_env import GymGazeboEnv
from gym.envs.registration import register
from std_msgs.msg import Float64
from gazebo_msgs.msg import LinkStates, ModelStates
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError

register(
  id='DoorOpenTash-v0',
  entry_point='envs.door_open_task_env:DoorOpenTaskEnv')

class DoorOpenTaskEnv(GymGazeboEnv):

  def __init__(self):
    """
    Initializes a new DoorOpenTaskEnv environment.
    """
    super(DoorOpenTaskEnv, self).__init__(
      start_init_physics_parameters=False,
      reset_world_or_sim="SIMULATION"
    )

    rospy.logdebug("Start DoorOpenTaskEnv INIT...")
    self.gazebo.unpauseSim()
    self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    self.robot_pos_sub = rospy.Publisher('gazebo/model_states', ModelStates, self._robot_pose_cb)
    self.image_sub = rospy.Subscriber('/front_cam/image_raw', Image, self._image_cb)
    self.robot_pose = Pose()
    self.bridge = CvBridge()
    self.image = Image()
    self.camera_ready = False
    self.high_action = np.array([1.0,1.0])
    self.low_action = np.array([-1.0,-1.0])
    self.action_space = spaces.Box(low=self.low_action, high=self.high_action)
    self.info = {}
    self._check_all_sensors_ready()
    self.gazebo.pauseSim()
    rospy.logdebug("Finished DoorOpenTaskEnv INIT...")

  def _robot_pose_cb(self,data):
      index = data.name.indexOf('mobile_robot')
      self.robot_pose = data.pose[index]

  def _image_cb(self, data):
      try:
          self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
          self.image = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
          self.camera_ready = True
      except CvBridgeError as e:
          self.camera_ready = False
          print(e)

  def _set_init(self):
    """Sets the Robot in its init pose
    """
    return

  def _check_all_systems_ready(self):
    """
    Checks that all the sensors, publishers and other simulation systems are
    operational.
    """
    return self._check_all_sensors_ready()

  def _check_all_sensors_ready(self):
    return self.camera_ready

  def _get_observation(self):
    return self.image

  def _post_information(self):
    """Returns the info.
    """
    return self.robot_pose

  def _take_action(self, action):
    """Applies the given action to the simulation.
    """
    msg = Twist()
    msg.linear.x = action[0]
    msg.linear.y = 0
    msg.linear.z = 0
    msg.angular.x = 0
    msg.angular.y = 0
    msg.angular.z = action[1]
    self.vel_pub.publish(msg)

  def _is_done(self):
    """Checks if episode done based on observations given.
    """
    if self.robot_pose.position.x < -0.5:
        return True
    return False

  def _compute_reward(self):
    """Calculates the reward to give based on the observations given.
    """
    if self.robot_pose.position.x < -0.5:
        reward = 100
    else:
        reward = 0
    return reward

  def _env_setup(self, initial_qpos):
    """Initial configuration of the environment. Can be used to configure initial state
    and extract information from the simulation.
    """
    return
