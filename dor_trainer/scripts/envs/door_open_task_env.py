#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import numpy as np
import rospy
from .gym_gazebo_env import GymGazeboEnv
from gym.envs.registration import register
from std_msgs.msg import Float64
from gazebo_msgs.msg import LinkStates, ModelStates, ModelState
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose, Twist
from sensor_msgs.msg import Image
from tf.transformations import quaternion_from_euler
import cv2
from cv_bridge import CvBridge, CvBridgeError

register(
  id='DoorOpenTash-v0',
  entry_point='envs.door_open_task_env:DoorOpenTaskEnv')

class DoorOpenTaskEnv(GymGazeboEnv):

  def __init__(self,resolution=(64,64)):
    """
    Initializes a new DoorOpenTaskEnv environment.
    """
    super(DoorOpenTaskEnv, self).__init__(
      start_init_physics_parameters=False,
      reset_world_or_sim="WORLD"
    )

    rospy.logdebug("Start DoorOpenTaskEnv INIT...")
    self.gazebo.unpauseSim()

    self.out = False
    self.bridge = CvBridge()
    self.resolution = resolution
    self.max_episode_steps = 120
    self.info = {}
    self.action_space = 2.0*np.array([[-1.0,-1.0],[-1.0,0.0],[-1.0,1.0],[0.0,-1.0],[0.0,0.0],[0.0,1.0],[1.0,-1.0],[1.0,0.0],[1.0,1.0]]) # x and yaw velocities
    self._check_all_sensors_ready()
    self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    self.model_state_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
    self.robot_pos_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self._robot_pose_cb)
    self.image_sub = rospy.Subscriber('/front_cam/image_raw', Image, self._image_cb)
    self._check_publisher_connection()

    self.gazebo.pauseSim()
    rospy.logdebug("Finished DoorOpenTaskEnv INIT...")

  def _robot_pose_cb(self,data):
      index = data.name.index('mobile_robot')
      self.robot_pose = data.pose[index]

  def _image_cb(self, data):
      try:
          self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
          # show image
          cv2.namedWindow("door opening")
          cv2.imshow('door opening',self.image)
          cv2.waitKey(1)
      except CvBridgeError as e:
          print(e)

  def _set_init(self):
    """Sets the Robot in its init pose
    """
    self.out = False
    robot = ModelState()
    robot.model_name = 'mobile_robot'
    robot.pose.position.x = 0.61
    robot.pose.position.y = 0.77
    robot.pose.position.z =0.075
    q = quaternion_from_euler(0,0,3.3)
    robot.pose.orientation.x = q[0]
    robot.pose.orientation.y = q[1]
    robot.pose.orientation.z = q[2]
    robot.pose.orientation.w = q[3]
    self.model_state_pub.publish(robot)

  def _check_all_systems_ready(self):
    """
    Checks that all the sensors, publishers and other simulation systems are
    operational.
    """
    self._check_all_sensors_ready()

  def _check_all_sensors_ready(self):
    self.image = None
    rospy.logdebug("Waiting for /front_cam/image_raw to be READY...")
    while self.image is None and not rospy.is_shutdown():
      try:
        data = rospy.wait_for_message("/front_cam/image_raw", Image, timeout=5.0)
        self.image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        rospy.logdebug("Current /front_cam/image_raw READY=>")
      except:
        rospy.logerr("Current /front_cam/image_raw not ready yet, retrying for getting /front_cam/image_raw")

    self.robot_pose = None
    rospy.logdebug("Waiting for /gazebo/model_states to be READY...")
    while self.robot_pose is None and not rospy.is_shutdown():
      try:
        data = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=5.0)
        index = data.name.index('mobile_robot')
        self.robot_pose = data.pose[index]
        rospy.logdebug("Current  /gazebo/model_statesREADY=>")
      except:
        rospy.logerr("Current  /gazebo/model_states not ready yet, retrying for getting  /gazebo/model_states")

  def _check_publisher_connection(self):
    """
    Checks that all the publishers are working
    :return:
    """
    rate = rospy.Rate(10)  # 10hz
    while self.vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
      rospy.logdebug("No susbribers to vel_pub yet so we wait and try again")
      try:
        rate.sleep()
      except rospy.ROSInterruptException:
        # This is to avoid error when world is rested, time when backwards.
        pass
    rospy.logdebug("vel_pub Publisher Connected")
    rospy.logdebug("All Publishers READY")


  def _get_observation(self):
    img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, self.resolution)
    float_arr = np.array(img)/255
    obs = float_arr.reshape((64,64,1))
    return obs

  def _post_information(self):
    """Returns the info.
    """
    return self.robot_pose

  def _take_action(self, action_idx):
    """Applies the given action to the simulation.
    """
    action = self.action_space[action_idx]
    msg = Twist()
    msg.linear.x = action[0]
    msg.linear.y = 0
    msg.linear.z = 0
    msg.angular.x = 0
    msg.angular.y = 0
    msg.angular.z = action[1]
    self.vel_pub.publish(msg)
    rospy.sleep(0.1)

  def _is_done(self):
    """Checks if episode done based on observations given.
    """
    if self.robot_pose.position.x < -0.5:
        self.out = True
        return True
    elif self.robot_pose.position.x > 1.5:
        return True
    return False

  def _compute_reward(self):
    """Calculates the reward to give based on the observations given.
    """
    if self.robot_pose.position.x < -0.5:
        return 100
    else:
        return -0.1
