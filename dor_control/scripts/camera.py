#!/usr/bin/env python
import rospy
import numpy as np
import cv2 as cv
from cv_bridge import CvBridge, CvBridgeError

from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import sensor_msgs.point_cloud2 as pc2

# realsense d435
class RSD435:
    # create a image view with a frame size for the ROI
    def __init__(self):
        print("create realsense d435 instance...")
        self.bridge=CvBridge()
        # camera information
        self.cameraInfoUpdate = False
        self.intrinsic = None
        # ros-realsense
        self.caminfo_sub = rospy.Subscriber('/rs435/color/camera_info', CameraInfo, self._caminfo_callback)
        self.depth_sub = rospy.Subscriber('/rs435/depth/image_raw', Image, self._depth_callback)
        self.color_sub = rospy.Subscriber('/rs435/color/image_raw', Image, self._color_callback)

        # data
        self.cv_color = []
        self.cv_depth = []
        self.width = 1024
        self.height = 720

    def ready(self):
        return self.cameraInfoUpdate and len(self.cv_color) > 0 and len(self.cv_depth) > 0

    def image_size(self):
        return self.height, self.width

    #### depth info
    # calculate mean distance in a small pixel frame around u,v
    # a non-zero mean value for the pixel with its neighboring pixels
    def distance(self,u,v,size=3):
        dist_list=[]
        for i in range(-size,size):
            for j in range(-size,size):
                value = self.cv_depth[v+j,u+i]
                if value > 0.0:
                    dist_list.append(value)
        if not dist_list:
            return -1
        else:
            return np.mean(dist_list)

    #### find 3d point with pixel and depth information
    def point3d(self,u,v):
        depth = self.distance(u,v)
        if depth < 0:
            return [-1,-1,-1]
        # focal length
        fx = self.intrinsic[0]
        fy = self.intrinsic[4]
        # principle point
        cx = self.intrinsic[2]
        cy = self.intrinsic[5]
        # deproject
        x = (u-cx)/fx
        y = (v-cy)/fy
        point3d = [depth*x/1000,depth*y/1000,depth/1000]
        return point3d

    #### data
    def depth_image(self):
        return self.cv_depth
    def color_image(self):
        return self.cv_color

    def _caminfo_callback(self, data):
        if self.cameraInfoUpdate == False:
            self.intrinsic = data.K
            self.width = data.width
            self.height = data.height
            self.cameraInfoUpdate = True

    def _depth_callback(self, data):
        if self.cameraInfoUpdate:
            try:
                self.cv_depth = self.bridge.imgmsg_to_cv2(data, data.encoding) #"16UC1"
            except CvBridgeError as e:
                print(e)

    def _color_callback(self, data):
        if self.cameraInfoUpdate:
            try:
                self.cv_color = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)
