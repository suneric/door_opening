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
from tf.transformations import quaternion_from_euler
import math
import cv2
from .sensors import CameraSensor, PoseSensor, ForceSensor


"""
Robot Driver
"""
class RobotDriver():
    def __init__(self,force_sensor):
        self.force_sensor = force_sensor
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
    # the robot dirver execute the command with a guassian distribution random error
    # with mean = 0, and stddev = 0.1 in linear and angular velocity
    def drive(self,vx,vyaw):
        #s = np.random.normal(1,0.1)
        #rate = rospy.Rate(20)
        # while not rospy.is_shutdown():
        s = self.safe_coefficient()
        #print("safe speed coefficient", s)
        msg = Twist()
        msg.linear.x = vx*s
        msg.linear.y = 0
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = vyaw*s
        self.vel_pub.publish(msg)
        # rate.sleep()

    def safe_coefficient(self):
        safe_max = 270 # N
        force = self.force_sensor.data();
        if force >= safe_max:
            return 0
        else:
            return 1-force/safe_max

    def stop(self):
        self.drive(0,0)

    def check_connection(self):
      rate = rospy.Rate(10)  # 10hz
      while self.vel_pub.get_num_connections() == 0 and not rospy.is_shutdown():
        rospy.logdebug("No susbribers to vel_pub yet so we wait and try again")
        try:
          rate.sleep()
        except rospy.ROSInterruptException:
          # This is to avoid error when world is rested, time when backwards.
          pass
      rospy.logdebug("vel_pub Publisher Connected")

###############################################################################
register(
  id='DoorOpenTash-v0',
  entry_point='envs.door_open_task_env:DoorOpenTaskEnv')

class DoorOpenTaskEnv(GymGazeboEnv):

  def __init__(self,resolution=(64,64),cam_noise=0.0):
    """
    Initializes a new DoorOpenTaskEnv environment, with define the image size
    and camera noise level (gaussian noise variance, the mean is 0.0)
    """
    super(DoorOpenTaskEnv, self).__init__(
      start_init_physics_parameters=False,
      reset_world_or_sim="WORLD"
    )

    self.door_dim = [0.9144, 0.0698] # length, width
    self.info = {}
    self.action_space = self._action_space()
    self.step_cnt = 0
    self.door_angle = 0.1 # inital angle of door
    self.success = False

    rospy.logdebug("Start DoorOpenTaskEnv INIT...")
    self.gazebo.unpauseSim()

    self.front_camera = CameraSensor(resolution,'/cam_front/image_raw',cam_noise)
    self.back_camera = CameraSensor(resolution,'/cam_back/image_raw',cam_noise)
    self.up_camera = CameraSensor(resolution,'/cam_up/image_raw',cam_noise)

    self.pose_sensor = PoseSensor()
    self.force_sensor = ForceSensor('/tf_sensor_topic')
    self.driver = RobotDriver(self.force_sensor)

    self._check_all_sensors_ready()
    self.robot_pose_pub = rospy.Publisher('/gazebo/set_model_state', ModelState, queue_size=1)
    self._check_publisher_connection()

    self.gazebo.pauseSim()
    rospy.logdebug("Finished DoorOpenTaskEnv INIT...")

  # a safe velocity of robot is 1.5 m/s, given a radius of 0.5 m robot, the safe angular
  # velocity can be pi rad/s
  def _action_space(self):
      lv,av = 1.5,3.14
      safe_base = np.array([[lv,av],[lv,0],[0,av],[-lv,av],[-lv,0],[lv,-av],[0,-av],[-lv,-av]])
      action_space = 2*safe_base
      print("action space", action_space)
      return action_space

  def action_dimension(self):
      dim = self.action_space.shape
      print("action space dimension", dim[0])
      return dim[0]

  def _check_all_systems_ready(self):
    """
    Checks that all the sensors, publishers and other simulation systems are
    operational.
    """
    self._check_all_sensors_ready()
    self._check_publisher_connection()

  def _check_all_sensors_ready(self):
    self.front_camera.check_camera_ready()
    self.back_camera.check_camera_ready()
    self.up_camera.check_camera_ready()
    self.pose_sensor.check_sensor_ready()
    self.force_sensor.check_force_sensor_ready()
    rospy.logdebug("All Sensors READY")

  def _check_publisher_connection(self):
    self.driver.check_connection()
    rospy.logdebug("All Publishers READY")

  def _get_observation(self):
    self._display_images()
    img_front = self.front_camera.grey_arr()
    img_back = self.back_camera.grey_arr()
    img_up = self.up_camera.grey_arr()
    # (64x64x3)
    obs = np.concatenate((img_front,img_back,img_up),axis=2)
    # print(obs.shape)
    return obs

  def _display_images(self):
      front = self.front_camera.rgb_image
      back = self.back_camera.rgb_image
      up = self.up_camera.rgb_image
      img = np.concatenate((front,back,up),axis=1)
      cv2.namedWindow("front-back-up")
      img = cv2.resize(img, None, fx=0.5, fy=0.5)
      cv2.imshow('front-back-up',img)
      cv2.waitKey(3)

  # return the robot footprint and door position
  def _post_information(self):
      door_radius, door_angle = self._door_position()
      footprint_lf = self._robot_footprint_position(0.25,0.25)
      footprint_lr = self._robot_footprint_position(-0.25,0.25)
      footprint_rf = self._robot_footprint_position(0.25,-0.25)
      footprint_rr = self._robot_footprint_position(-0.25,-0.25)
      camera_pose = self._robot_footprint_position(0.5,-0.25)
      info = {}
      info['door'] = (door_radius,door_angle)
      info['robot'] = [(footprint_lf[0,3],footprint_lf[1,3]),
                    (footprint_rf[0,3],footprint_rf[1,3]),
                    (footprint_lr[0,3],footprint_lr[1,3]),
                    (footprint_rr[0,3],footprint_rr[1,3]),
                    (camera_pose[0,3], camera_pose[1,3])]
      return info

  #############################################################################
  # overidde functions
  def _set_init(self):
    raise NotImplementedError()

  def _take_action(self, action_idx):
    raise NotImplementedError()

  def _is_done(self):
    raise NotImplementedError()

  def _compute_reward(self):
    raise NotImplementedError()

  #############################################################################
  # utility functions
  def _reset_mobile_robot(self,x,y,z,yaw):
      robot = ModelState()
      robot.model_name = 'mobile_robot'
      robot.pose.position.x = x
      robot.pose.position.y = y
      robot.pose.position.z = z
      rq = quaternion_from_euler(0,0,yaw)
      robot.pose.orientation.x = rq[0]
      robot.pose.orientation.y = rq[1]
      robot.pose.orientation.z = rq[2]
      robot.pose.orientation.w = rq[3]
      self.robot_pose_pub.publish(robot)

  def _wait_door_closed(self):
      door_r, door_a = self._door_position()
      while door_a > 0.11:
          rospy.sleep(0.5)
          door_r, door_a = self._door_position()

  def _door_is_open(self):
      door_r, door_a = self._door_position()
      if door_a > 0.45*math.pi: # 81 degree
          return True
      else:
          return False

  # camera position in door polar coordinate frame
  # return radius to (0,0) and angle 0 for (0,1,0)
  def _camera_position(self):
      cam_pose = self._robot_footprint_position(0.5,-0.25)
      angle = math.atan2(cam_pose[0,3],cam_pose[1,3])
      radius = math.sqrt(cam_pose[0,3]*cam_pose[0,3]+cam_pose[1,3]*cam_pose[1,3])
      return radius, angle

  # door position in polar coordinate frame
  # retuen radius to (0,0) and angle 0 for (0,1,0)
  def _door_position(self):
      door_matrix = self._pose_matrix(self.pose_sensor.door())
      door_edge = np.array([[1,0,0,self.door_dim[0]],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]])
      door_edge_mat = np.dot(door_matrix, door_edge)
      # open angle [0, pi/2]
      open_angle = math.atan2(door_edge_mat[0,3],door_edge_mat[1,3])
      return self.door_dim[0], open_angle


  # robot is out of the door way (x < 0)
  def _robot_is_out(self):
      # footprint of robot
      fp_lf = self._robot_footprint_position(0.25,0.25)
      fp_lr = self._robot_footprint_position(-0.25,0.25)
      fp_rf = self._robot_footprint_position(0.25,-0.25)
      fp_rr = self._robot_footprint_position(-0.25,-0.25)
      cam_p = self._robot_footprint_position(0.5,-0.25)
      if fp_lf[0,3] < 0.0 and fp_lr[0,3] < 0.0 and fp_rf[0,3] < 0.0 and fp_rr[0,3] < 0.0 and cam_p[0,3] < 0.0:
          return True
      else:
          return False

  def _robot_is_in(self):
      # footprint of robot
      dr, da = self._door_position()
      fp_lf = self._robot_footprint_position(0.25,0.25)
      fp_lr = self._robot_footprint_position(-0.25,0.25)
      fp_rf = self._robot_footprint_position(0.25,-0.25)
      fp_rr = self._robot_footprint_position(-0.25,-0.25)
      cam_p = self._robot_footprint_position(0.5,-0.25)
      d_x = dr*math.sin(da)
      if fp_lf[0,3] > d_x and fp_lr[0,3] > d_x and fp_rf[0,3] > d_x and fp_rr[0,3] > d_x and cam_p[0,3] > d_x:
          return True
      else:
          return False

  # utility function
  def _robot_footprint_position(self,x,y):
      robot_matrix = self._pose_matrix(self.pose_sensor.robot())
      footprint_trans = np.array([[1,0,0,x],
                               [0,1,0,y],
                               [0,0,1,0],
                               [0,0,0,1]])
      fp_mat = np.dot(robot_matrix, footprint_trans)
      return fp_mat

  def _pose_matrix(self, cp):
    position = cp.position
    orientation = cp.orientation
    matrix = np.eye(4)
    # translation
    matrix[0,3] = position.x# in meter
    matrix[1,3] = position.y
    matrix[2,3] = position.z
    # quaternion to matrix
    x = orientation.x
    y = orientation.y
    z = orientation.z
    w = orientation.w

    Nq = w*w + x*x + y*y + z*z
    if Nq < 0.001:
        return matrix

    s = 2.0/Nq
    X = x*s
    Y = y*s
    Z = z*s
    wX = w*X; wY = w*Y; wZ = w*Z
    xX = x*X; xY = x*Y; xZ = x*Z
    yY = y*Y; yZ = y*Z; zZ = z*Z
    matrix=np.array([[1.0-(yY+zZ), xY-wZ, xZ+wY, position.x],
            [xY+wZ, 1.0-(xX+zZ), yZ-wX, position.y],
            [xZ-wY, yZ+wX, 1.0-(xX+yY), position.z],
            [0, 0, 0, 1]])
    return matrix
