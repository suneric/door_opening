#!/usr/bin/env python
import rospy
import numpy as np
import actionlib
from std_msgs.msg import Float64
from actionlib_msgs.msg import GoalStatus
from geometry_msgs.msg import Pose, Twist, PoseWithCovarianceStamped
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

# base class of move task
class MoveTask:
    def __init__(self,goal):
        self.goal = goal
        self.goal_status = 'ready' # moving, reached
        self.task_status = 'inprogress' # 'done'

        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()
        self.last_time = rospy.Time.now()

    def move_status(self):
        return self.goal_status

    def is_done(self):
        if self.task_status == 'done':
            return True
        else:
            return False

    def reach_cb(self,msg,result):
        # print('msg: ', msg, 'result: ', result)
        if msg == GoalStatus.SUCCEEDED: # 3
            self.goal_status = 'reached'
        else:
            print("update path plan: ")
            self.move()

    def moving_cb(self):
        self.goal_status = "moving"

    def feedback_cb(self, feedback):
        # reset the goal every minute
        # print(feedback)
        current_time = rospy.Time.now()
        duration = current_time.secs - self.last_time.secs
        if duration > 60:
            print("update path plan: ")
            self.move()

    def move(self):
        self.last_time = rospy.Time.now()
        goal = MoveBaseGoal()
        goal.target_pose.header.frame_id = "map"
        goal.target_pose.header.stamp = self.last_time
        goal.target_pose.pose = self.goal
        self.client.send_goal(goal, self.reach_cb, self.moving_cb, self.feedback_cb)
        self.goal_status = 'moving'
        rospy.loginfo(self.goal)

class ChargeTask(MoveTask):
    def __init__(self,pos):
        MoveTask.__init__(self,pos)

    def perform(self):
        status = self.move_status()
        if status == 'ready':
            rospy.loginfo("move to charge...")
            self.move()

        if status == 'reached':
            self.task_status = 'done'


class PushDoorTask(MoveTask):
    def __init__(self,pos):
        MoveTask.__init__(self,pos)
        self.robot_pos = 'in' #'out'
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)
        self.hook_pub = rospy.Publisher('mobile_robot/joint_hook_controller/command', Float64, queue_size=1)
        self.pose_sub = rospy.Subscriber('amcl_pose', PoseWithCovarianceStamped, self.amcl_cb)
    def amcl_cb(self,data):
        #print(data)
        amcl_x = data.pose.pose.position.x
        if amcl_x < -1.0:
            self.robot_pos = 'out'

    def push_door(self):
        self.hook_pub.publish(1.57)
        vel_msg = Twist()
        vel_msg.linear.x = 2.0 # forward
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        self.vel_pub.publish(vel_msg)

    def stop(self):
        self.hook_pub.publish(0.0)
        vel_msg = Twist()
        vel_msg.linear.x = 2.0 # forward
        vel_msg.linear.y = 0
        vel_msg.linear.z = 0
        vel_msg.angular.x = 0
        vel_msg.angular.y = 0
        vel_msg.angular.z = 0
        self.vel_pub.publish(vel_msg)

    def perform(self):
        status = self.move_status()
        if status == 'ready':
            rospy.loginfo("move to push door...")
            self.move()

        if status == 'reached':
            if self.robot_pos != 'out':
                self.push_door()
                self.task_status = 'inprogress'
            else:
                self.stop()
                self.task_status = 'done'


class PullDoorTask(MoveTask):
    def __init__(self, pos):
        MoveTask.__init__(self,pos)
        self.vel_pub = rospy.Publisher('cmd_vel', Twist, queue_size=1)

    def perform(self):
        status = self.move_status()
        if status == 'ready':
            rospy.loginfo("move to pull door...")
            self.move()

        if status == 'reached':
            self.task_status = 'done'

class DisinfectTask(MoveTask):
    def __init__(self, pos):
        MoveTask.__init__(self,pos)

    def perform(self):
        status = self.move_status()
        if status == 'ready':
            rospy.loginfo("move to disinfect...")
            rospy.loginfo(self.goal)
            self.move()

        if status == 'reached':
            self.task_status = 'done'

# positions
def get_charge_pose():
    pose = Pose()
    pose.position.x = 8.0
    pose.position.y = 1.0
    pose.position.z = 0.0
    pose.orientation.x = 0
    pose.orientation.y = 0
    pose.orientation.z = 0
    pose.orientation.w = 1
    return pose

def get_disinfection_pose():
    pose = Pose()
    pose.position.x = 1.5
    pose.position.y = 6.0
    pose.position.z = 0.0
    pose.orientation.x = 0
    pose.orientation.y = 0
    pose.orientation.z = 0
    pose.orientation.w = 1
    return pose

def get_door_push_pose():
    pose = Pose()
    pose.position.x = 1.5
    pose.position.y = 5.7
    pose.position.z = 0.0
    pose.orientation.x = 0
    pose.orientation.y = 0
    pose.orientation.z = 1
    pose.orientation.w = 0
    return pose

def get_door_pull_pose():
    pose = Pose()
    pose.position.x = 1.0
    pose.position.y = 0.7
    pose.position.z = 0.0
    pose.orientation.x = 0
    pose.orientation.y = 0
    pose.orientation.z = 0
    pose.orientation.w = 1
    return pose

if __name__ == '__main__':
    rospy.init_node("task_executor", anonymous=True, log_level=rospy.INFO)
    rate = rospy.Rate(10)
    task1 = ChargeTask(get_charge_pose())
    task2 = DisinfectTask(get_disinfection_pose())
    task3 = PushDoorTask(get_door_push_pose())
    task4 = PullDoorTask(get_door_pull_pose())
    try:
        while not rospy.is_shutdown():
            if not task1.is_done():
                task1.perform()
            else:
                if not task2.is_done():
                    task2.perform()
                else:
                    if not task3.is_done():
                        task3.perform()
                    else:
                        if not task4.is_done():
                            task4.perform()
                        else:
                            rospy.logininf("all tasks are performed.")
        rate.sleep()
    except rospy.ROSInterruptException:
        pass
