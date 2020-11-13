#!/usr/bin/env python

import numpy as np
import rospy
from gazebo_msgs.msg import LinkStates, ModelStates, ModelState, LinkState
from geometry_msgs.msg import Pose, Twist, WrenchStamped
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import math
import skimage

"""
ForceSensor for the sidebar tip hook joint
"""
class ForceSensor():
    def __init__(self, topic='/tf_sensor_topic'):
        self.topic=topic
        self.force_sub = rospy.Subscriber(self.topic, WrenchStamped, self._force_cb)
        self.max_force = 0.0
        self.record = []

    def _force_cb(self,data):
        force = data.wrench.force
        max = np.amax([force.x,force.y,force.z])
        if self.max_force < max:
            self.max_force = max

    # get sensored force data in x,y,z direction
    def data(self):
        max_force = self.max_force
        self.record.append(max_force)
        self.max_force = 0.0
        return max_force

    def force_record(self):
        return self.record

    def reset_record(self):
        self.record = []

    def check_force_sensor_ready(self):
        self.force_data = None
        while self.force_data is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message(self.topic, WrenchStamped, timeout=5.0)
                self.force_data = data.wrench.force
                rospy.logdebug("Current force sensor READY=>")
            except:
                rospy.logerr("Current force sensor not ready yet, retrying for getting force info")

"""
CameraSensor with resolution, topic and guassian noise level by default noise(stddev**2) = 0.02, mean = 0.0
"""
class CameraSensor():
    def __init__(self, resolution=(64,64), topic='/cam_front/image_raw', noise=0):
        self.resolution = resolution
        self.topic = topic
        self.noise = noise
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(self.topic, Image, self._image_cb)
        self.rgb_image = None
        self.grey_image = None

    def _image_cb(self,data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.rgb_image = self._guass_noisy(image, self.noise)
            # print("image", image, "noise", self.rgb_image)
            self.grey_image = cv2.cvtColor(self.rgb_image, cv2.COLOR_BGR2GRAY)
        except CvBridgeError as e:
            print(e)

    def check_camera_ready(self):
        self.rgb_image = None
        while self.rgb_image is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message(self.topic, Image, timeout=5.0)
                image = self.bridge.imgmsg_to_cv2(data, "bgr8")
                self.rgb_image = self._guass_noisy(image, self.noise)
                rospy.logdebug("Current image READY=>")
            except:
                rospy.logerr("Current image not ready yet, retrying for getting image")

    def image_arr(self):
        img = cv2.resize(self.rgb_image, self.resolution)
        # normalize the image for easier training
        img_arr = np.array(img)/255 - 0.5
        img_arr = img_arr.reshape((64,64,3))
        return img_arr

    def grey_arr(self):
        img = cv2.resize(self.grey_image, self.resolution)
        # normalize the image for easier training
        img_arr = np.array(img)/255 - 0.5
        img_arr = img_arr.reshape((64,64,1))
        return img_arr

    def _guass_noisy(self,image,var):
        if var == 0.0:
            return image
        img = skimage.util.img_as_float(image)
        noisy = skimage.util.random_noise(img,'gaussian',mean=0.0,var=var)
        return skimage.util.img_as_ubyte(noisy)

    def _noisy(self,image,type):
        if type == 'guassian':
            img = skimage.util.img_as_float(image)
            noisy = skimage.util.random_noise(img,'gaussian')
            res = skimage.util.img_as_ubyte(noisy)
        elif type == 'salt':
            img = skimage.util.img_as_float(image)
            noisy = skimage.util.random_noise(img,'salt')
            res = skimage.util.img_as_ubyte(noisy)
        elif type == 'pepper':
            img = skimage.util.img_as_float(image)
            noisy = skimage.util.random_noise(img,'pepper')
            res = skimage.util.img_as_ubyte(noisy)
        elif type == 'poisson':
            img = skimage.util.img_as_float(image)
            noisy = skimage.util.random_noise(img,'poisson')
            res = skimage.util.img_as_ubyte(noisy)
        else:
            res = image
        return res

"""
pose sensor
"""
class PoseSensor():
    def __init__(self, noise=0.0):
        self.noise = noise
        self.door_pose_sub = rospy.Subscriber('/gazebo/link_states', LinkStates, self._door_pose_cb)
        self.robot_pos_sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self._robot_pose_cb)
        self.robot_pose = None
        self.door_pose = None

    def _door_pose_cb(self,data):
        index = data.name.index('hinged_door::door')
        self.door_pose = data.pose[index]

    def _robot_pose_cb(self,data):
        index = data.name.index('mobile_robot')
        self.robot_pose = data.pose[index]

    def robot(self):
        # add noise
        return self.robot_pose

    def door(self):
        return self.door_pose

    def check_sensor_ready(self):
        self.robot_pose = None
        rospy.logdebug("Waiting for /gazebo/model_states to be READY...")
        while self.robot_pose is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/gazebo/model_states", ModelStates, timeout=5.0)
                index = data.name.index('mobile_robot')
                self.robot_pose = data.pose[index]
                rospy.logdebug("Current  /gazebo/model_states READY=>")
            except:
                rospy.logerr("Current  /gazebo/model_states not ready yet, retrying for getting  /gazebo/model_states")

        self.door_pose = None
        rospy.logdebug("Waiting for /gazebo/link_states to be READY...")
        while self.door_pose is None and not rospy.is_shutdown():
            try:
                data = rospy.wait_for_message("/gazebo/link_states", LinkStates, timeout=5.0)
                index = data.name.index('hinged_door::door')
                self.door_pose = data.pose[index]
                rospy.logdebug("Current  /gazebo/link_states READY=>")
            except:
                rospy.logerr("Current  /gazebo/link_states not ready yet, retrying for getting  /gazebo/link_states")
