#!/usr/bin/env python
import rospy
import os
import sys

from geometry_msgs.msg import Twist

class AutoMove():
    def __init__(self):
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.task_status = 'ready' # inprograss, done

    def explore(self):
        self.task_status = 'inprogress'
        rospy.sleep(5)
        self.drive_forward(2.0)
        rospy.sleep(40)
        self.drive_rotation(2.0)
        rospy.sleep(26)
        self.drive_forward(2.0)
        rospy.sleep(33)
        self.drive_rotation(2.0)
        rospy.sleep(25.5)
        self.drive_forward(2.0)
        rospy.sleep(55)
        self.drive_rotation(-2.0)
        rospy.sleep(26)
        self.drive_forward(1.0)
        rospy.sleep(20)
        self.drive_rotation(2.0)
        rospy.sleep(52)
        self.drive_forward(1.0)
        rospy.sleep(100)
        self.stop_drive()
        self.task_status = 'done'

    def drive_forward(self,v=1.0):
        self.vel_pub.publish(self.vel_msg(v,0))

    def drive_rotation(self,v=1.0):
        self.vel_pub.publish(self.vel_msg(0,v))

    def stop_drive(self):
        self.vel_pub.publish(self.vel_msg(0,0))

    def vel_msg(self,vx=1.0,vr=1.0):
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = 0
        msg.linear.z = 0
        msg.angular.x = 0
        msg.angular.y = 0
        msg.angular.z = vr
        return msg

    def status(self):
        return self.task_status

if __name__ == '__main__':
    rospy.init_node('auto_move', anonymous=True, log_level=rospy.INFO)
    auto = AutoMove()
    rate = rospy.Rate(10)
    try:
        while not rospy.is_shutdown():
            if auto.status() == 'ready':
                auto.explore()
            elif auto.status == 'done':
                rospy.loginfo("exploration is done")
            rate.sleep()
    except rospy.ROSInterruptException:
        pass
