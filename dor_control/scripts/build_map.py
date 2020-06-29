#!/usr/bin/env python
import rospy
import os
import sys

from geometry_msgs.msg import Twist

class AutoMove():
    def __init__(self):
        self.publisher = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.x = 0.0
        # self.y = 0.0
        # self.z = 0.0
        self.th = 0.0
        self.speed = 0.0
        self.turn = 0.0

    def set_vel(self,x,th,speed,turn):
        self.x = x
        # self.y = y
        # self.z = z
        self.th = th
        self.speed = speed
        self.turn = turn

    def step(self):
        twist = Twist()
        twist.linear.x = self.x * self.speed
        # twist.linear.y = self.y * self.speed
        # twist.linear.z = self.z * self.speed
        # twist.angular.x = 0
        # twist.angular.y = 0
        twist.angular.z = self.th * self.turn
        self.publisher.publish(twist)

if __name__ == '__main__':
    rospy.init_node('auto_move', anonymous=True, log_level=rospy.INFO)
    speed = 0.5
    turn = 2.0
    auto = AutoMove()
    auto.set_vel(1.0,1.0,speed,turn)
    rate = rospy.Rate(10)
    try:
        while not rospy.is_shutdown():
            auto.step()
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
