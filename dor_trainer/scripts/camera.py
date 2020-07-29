#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo, PointCloud2

class FrontCam:
    def __init__(self):
        self.bridge=CvBridge()
        # camera information
        self.cameraInfoUpdate = False
        # ros-realsense
        self.caminfo_sub = rospy.Subscriber('/front_cam/camera_info', CameraInfo, self._caminfo_callback)
        self.color_sub = rospy.Subscriber('/front_cam/image_raw', Image, self._color_callback)
        # data
        self.cv_color = []
        self.width = 1024
        self.height = 720

    def ready(self):
        return self.cameraInfoUpdate and len(self.cv_color) > 0

    def image_size(self):
        return self.height, self.width

    def color_image(self):
        return self.cv_color

    def _caminfo_callback(self, data):
        if self.cameraInfoUpdate == False:
            self.width = data.width
            self.height = data.height
            self.cameraInfoUpdate = True

    def _color_callback(self, data):
        if self.cameraInfoUpdate:
            try:
                self.cv_color = self.bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)

    def draw_image(self):
        img = sensor.color_image()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.namedWindow("door opening")
        img = cv2.resize(img, None, fx=1.0, fy=1.0)
        cv2.imshow('door opening',img)
        cv2.waitKey(3)

if __name__ == '__main__':
    rospy.init_node("camera_door_open", anonymous=True, log_level=rospy.INFO)
    sensor = FrontCam()
    rate = rospy.Rate(10)
    try:
        while not rospy.is_shutdown():
            if sensor.ready():
                sensor.draw_image()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
    cv2.destroyAllWindows()
