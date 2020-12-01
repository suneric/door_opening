#!/usr/bin/env python

from __future__ import absolute_import, division, print_function

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from .sensors import ForceSensor


"""
Robot Driver
"""
class RobotDriver():
    def __init__(self,safe_max=270):
        self.safe_max = safe_max
        self.force_sensor = ForceSensor('/tf_sensor_topic')
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

    # drive with constant speed
    def drive(self,vx,vyaw):
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = 0
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = vyaw
        self.vel_pub.publish(msg)

    # drive with changable speed
    def drive_vs(self,vx,vyaw):
        s = self.safe_coefficient()
        msg = Twist()
        msg.linear.x = vx*s
        msg.linear.y = 0
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = vyaw*s
        self.vel_pub.publish(msg)
        return s

    def safe_coefficient(self):
        force = self.force_sensor.data();
        if force >= self.safe_max:
            return 0
        else:
            return 1-force/self.safe_max

    def stop(self):
        self.drive(0,0)

    def check_driver_ready(self):
        self.force_sensor.check_force_sensor_ready()

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
