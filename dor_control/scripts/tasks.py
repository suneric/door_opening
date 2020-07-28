#!/usr/bin/env python
import rospy
import numpy as np
import actionlib
from std_msgs.msg import Float64
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, Twist, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from tf.transformations import quaternion_from_euler
from std_msgs.msg import String

# base class of move task
class MoveTask:
    def __init__(self,goal):
        self.goal = goal
        self.goal_status = 'ready' # moving, reached
        self.task_status = 'ready' # inprogress, done
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.pose_sub = rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, self.amcl_cb)
        self.amcl_pose = goal.position
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        self.last_time = rospy.Time.now()

    def is_done(self):
        if self.task_status == 'done' or self.task_status == 'ready':
            return True
        else:
            return False

    def amcl_cb(self,data):
        self.amcl_pose = data.pose.pose.position

    def reach_cb(self,msg,result):
        if msg == GoalStatus.SUCCEEDED: # 3
            self.goal_status = 'reached'
        else:
            print("update path plan: ")
            self.auto_move()

    def moving_cb(self):
        self.goal_status = "moving"

    def feedback_cb(self, feedback):
        # reset the goal every minute
        # print(feedback)
        current_time = rospy.Time.now()
        duration = current_time.secs - self.last_time.secs
        if duration > 120:
            print("update path plan: ")
            self.auto_move()

    def auto_move(self):
        self.last_time = rospy.Time.now()
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = self.last_time
        goal.target_pose.pose = self.goal
        self.client.send_goal(goal, self.reach_cb, self.moving_cb, self.feedback_cb)
        self.goal_status = 'moving'
        rospy.loginfo("autonomously moving to ")
        rospy.loginfo(self.goal)

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

# self-charing task
class ChargeTask(MoveTask):
    def __init__(self,pos):
        MoveTask.__init__(self,pos)
        self.vslider_pub = rospy.Publisher('mobile_robot/joint_vertical_controller/command', Float64, queue_size=1)
        self.hslider_pub = rospy.Publisher('mobile_robot/joint_horizontal_controller/command', Float64, queue_size=1)
        self.walloutlet_found = False
        self.detection_sub = rospy.Subscriber('detection/walloutlet', String, self.detection_cb)

    def detection_cb(self,msg):
        if msg.data == "found":
            self.walloutlet_found = True

    def auto_charge(self):
        if self.task_status == 'ready':
            self.task_status = 'inprogress'
            rospy.loginfo("approaching to walloutlet...")
            self.vslider_pub.publish(0.05)
            self.hslider_pub.publish(0.0)

        elif self.task_status == 'inprogress':
            if self.amcl_pose.x < 8.5:
                self.drive_forward()
            else:
                self.drive_forward(0.5)
                if self.walloutlet_found:
                    rospy.loginfo('found walloutlet')
                    self.stop_drive()
                    rospy.loginfo("auto charging (30 seconds)...")
                    rospy.sleep(30) # sleep 30 secs
                    self.task_status = 'done'
                    rospy.loginfo("auto charge done")
                elif self.amcl_pose.x > 9.3:
                    rospy.loginfo('too close to the wall')
                    self.stop_drive()
                    self.task_status = 'done'
        else:
            rospy.loginfo("auto charge done")

    def perform(self):
        if self.goal_status == 'ready':
            self.auto_move()
        if self.goal_status == 'reached':
            self.auto_charge()

# disinfection task
class DisinfectTask(MoveTask):
    def __init__(self, pos):
        MoveTask.__init__(self,pos)
        self.disinfect_time = rospy.Time.now()

    def disinfect(self):
        if self.task_status == 'ready':
            self.task_status = 'inprogress'
            rospy.loginfo("disinfecting (30 seconds)...")
            self.disinfect_time = rospy.Time.now()
        elif self.task_status == 'inprogress':
            current_time = rospy.Time.now()
            duration = current_time.secs - self.disinfect_time.secs
            if duration < 30:
                self.drive_rotation(1.0)
            else:
                self.stop_drive()
                self.task_status = 'done'
                rospy.loginfo("disinfection done")
        else:
            rospy.loginfo("disinfection done")

    def perform(self):
        if self.goal_status == 'ready':
            self.auto_move()
        if self.goal_status == 'reached':
            self.disinfect()

# push door task
class PushDoorTask(MoveTask):
    def __init__(self,pos):
        MoveTask.__init__(self,pos)
        self.hook_pub = rospy.Publisher('mobile_robot/joint_hook_controller/command', Float64, queue_size=1)
        self.vslider_pub = rospy.Publisher('mobile_robot/joint_vertical_controller/command', Float64, queue_size=1)
        self.hslider_pub = rospy.Publisher('mobile_robot/joint_horizontal_controller/command', Float64, queue_size=1)
        self.door_found = False
        self.detection_sub = rospy.Subscriber('detection/door_handle', String, self.detection_cb)

    def detection_cb(self,msg):
        if msg.data == 'found':
            self.door_found = True

    def push_door(self):
        if self.task_status == 'ready':
            self.task_status = 'inprogress'
            rospy.loginfo("pushing door...")
            self.vslider_pub.publish(1.0)
            self.hslider_pub.publish(0.0)
            self.hook_pub.publish(1.57)
        elif self.task_status == 'inprogress':
            if self.amcl_pose.x > -1.5:
                if self.door_found:
                    self.drive_forward(1.0)
                else:
                    rospy.loginfo("detecting door and door handle...")
                    self.drive_forward(0.5)
            else:
                self.stop_drive()
                self.hook_pub.publish(0.0)
                self.task_status = 'done'
                rospy.loginfo("push door done")
        else:
            rospy.loginfo("push door done")

    def perform(self):
        if self.goal_status == 'ready':
            self.auto_move()
        if self.goal_status == 'reached':
            self.push_door()

# pull door task
class PullDoorTask(MoveTask):
    def __init__(self,pos):
        MoveTask.__init__(self,pos)
        self.vslider_pub = rospy.Publisher('mobile_robot/joint_vertical_controller/command', Float64, queue_size=1)
        self.hslider_pub = rospy.Publisher('mobile_robot/joint_horizontal_controller/command', Float64, queue_size=1)
        self.hook_pub = rospy.Publisher('mobile_robot/joint_hook_controller/command', Float64, queue_size=1)
        self.task_status = 'inprogress'
        self.goal_status = 'reached'

    def open_door_1(self):
        self.stop_drive()
        rospy.sleep(2)
        self.vslider_pub.publish(0.88)
        rospy.sleep(2)
        self.drive_forward(-0.5)
        rospy.sleep(2)
        self.hook_pub.publish(1.57)
        rospy.sleep(2)
        self.drive_rotation(-1.0)
        rospy.sleep(2)
        self.vslider_pub.publish(1.0)
        self.hslider_pub.publish(-0.24)
        self.stop_drive()
    def open_door_2(self):
        for i in range(0,3):
            self.drive_rotation(-1.0)
            rospy.sleep(2)
            self.drive_forward(-1.0)
            rospy.sleep(1)
        for i in range(0,4):
            self.drive_rotation(-1.0)
            rospy.sleep(5)
            self.drive_forward(-1.0)
            rospy.sleep(1)
        for i in range(0,2):
            self.drive_rotation(-3.0)
            rospy.sleep(5)
            self.drive_forward(1.0)
            rospy.sleep(1.5)
        self.drive_rotation(-3.0)
        rospy.sleep(17)
        self.drive_forward(-2.0)
        rospy.sleep(12)
        self.hook_pub.publish(0.0)
        self.hslider_pub.publish(0.0)
        self.stop_drive()
        self.task_status = 'done'

    def pull_door(self):
        if self.task_status == 'ready':
            self.task_status = 'inprogress'
            rospy.loginfo("pulling door...")
        elif self.task_status == 'inprogress':
            # apprach the door
            self.vslider_pub.publish(1.0)
            self.hslider_pub.publish(0.0)
            # print(self.amcl_pose)
            if self.amcl_pose.x < -0.45:
                self.drive_forward(0.5)
            # elif self.amcl_pose.x > 2.0:
            #     self.stop_drive()
            #     self.task_status = 'done'
            #     rospy.loginfo("pull door done")
            else:
                self.open_door_1()
                self.open_door_2()

            # grab the door handle
            # pull the door
            # release the hook
            # rotate the robot using hook to hold the door and open the door

    def perform(self):
        if self.goal_status == 'ready':
            self.auto_move()
        if self.goal_status == 'reached':
            self.pull_door()

# positions
def task_pose(x,y,yaw):
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = 0.0
    q = quaternion_from_euler(0,0,yaw)
    pose.orientation.x = q[0]
    pose.orientation.y = q[1]
    pose.orientation.z = q[2]
    pose.orientation.w = q[3]
    return pose

if __name__ == '__main__':
    rospy.init_node("task_executor", anonymous=True, log_level=rospy.INFO)
    rate = rospy.Rate(10)
    task1 = ChargeTask(task_pose(7.5,1.0,0.0))
    task2 = DisinfectTask(task_pose(5.0,5.0,1.57))
    task3 = PushDoorTask(task_pose(2.5,5.5,3.14))
    task4 = PullDoorTask(task_pose(-1.0,0.8,0.0))
    try:
        while not rospy.is_shutdown():
            if not task4.is_done():
                task4.perform()
            else:
                rospy.loginfo("done")
            # if not task1.is_done():
            #     task1.perform()
            # else:
            #     if not task2.is_done():
            #         task2.perform()
            #     else:
            #         if not task3.is_done():
            #             task3.perform()
            #         else:
            #             if not task4.is_done():
            #                 task4.perform()
            #             else:
            #                 rospy.loginfo("all tasks are performed.")
        rate.sleep()
    except rospy.ROSInterruptException:
        pass
