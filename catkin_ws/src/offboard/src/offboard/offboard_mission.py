#!/usr/bin/env python

# Copyright 2020 Kenan Unal
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License
# for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

import rospy
import _thread
from threading import Thread
import time
import mavros

import math
import numpy as np
from mavros.utils import *
from mavros import setpoint  as SP
from mavros import mavlink
from mavros_msgs.msg import Mavlink,PositionTarget
from geometry_msgs.msg import PoseStamped, Quaternion,TwistStamped,Twist
from std_msgs.msg import Header

from pymavlink import mavutil
from tf.transformations import quaternion_from_euler
from offboard.mavros_helper import MavrosHelper
from six.moves import xrange

from sensor_msgs.msg import Image # Image is the message type
import cv2
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
from offboard_camera.msg import ObjectDetection

import random
from collections import deque

class OffBoardMission(MavrosHelper):
    def __init__(self):
        super().setup()
        self.pos = PoseStamped()
        self.last_position = PoseStamped()
        self.radius = 1
        self.waypoints = deque()
        self.last_yaw_degrees = 0
        # subscriber for camera
        self.camera_sub = rospy.Subscriber('navq/object_detection', ObjectDetection, self.process_image)
        
        # publisher for mavros/setpoint_position/local
        self.mavlink_pub = rospy.Publisher('mavlink/to', Mavlink, queue_size=1)
        self.pub = rospy.Publisher('mavros/setpoint_position/local', PoseStamped, queue_size=10)
        self.velocity_pub = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel', TwistStamped, queue_size=10)
        try:
            rospy.loginfo('Starting new thread')
            _thread.start_new_thread(self.navigate, ())
        except:
            raise Exception("Error: Unable to start thread")

        self.reached = False
        self.violation = False

        # need to simulate heartbeat to prevent datalink loss detection
        self.hb_mav_msg = mavutil.mavlink.MAVLink_heartbeat_message(
            mavutil.mavlink.MAV_TYPE_GCS, 0, 0, 0, 0, 0)
        self.hb_mav_msg.pack(mavutil.mavlink.MAVLink('', 2, 1))
        self.hb_ros_msg = mavlink.convert_to_rosmsg(self.hb_mav_msg)
        self.hb_thread = Thread(target=self.send_heartbeat, args=())
        self.hb_thread.daemon = True
        self.hb_thread.start()
        '''
        self.velocity = TwistStamped()
        self.velocity.header = Header()
        self.velocity.header.frame_id = "base_footprint"
        self.velocity.header.stamp = rospy.Time.now()
        self.velocity.twist = Twist()
        self.velocity_pub.publish(self.velocity)
        '''
    def process_image(self,data):
        #check if we still working on previous violation
        if self.violation:  
            rospy.logdebug("Violation check process is still in progress")
            return

        br = CvBridge()
        # Convert ROS Image message to OpenCV image
        current_frame = br.imgmsg_to_cv2(data.image)
        rate = rospy.Rate(20) 
        self.last_position = self.local_position
        self.violation = data.violation
        if (self.violation):
            self.waypoints.append([self.last_position.pose.position.x,self.last_position.pose.position.y, self.last_position.pose.position.z,self.last_yaw_degrees,5])
            self.waypoints.append([25,25,20,self.last_yaw_degrees,5])
          

    def check_violation(self, frame):
        return random.random() > 0.98

    def send_heartbeat(self):
        rate = rospy.Rate(2)  # Hz
        while not rospy.is_shutdown(): 
            try:  # prevent garbage in console output when thread is killed
                self.mavlink_pub.publish(self.hb_ros_msg)
                rate.sleep()
            except rospy.ROSException:
                pass

    def send_dummy_setPoint(self):
        rospy.loginfo("send few setpoint messages, then activate OFFBOARD mode, to take effect")
        rate = rospy.Rate(20)
        self.pos.header = Header()
        self.pos.header.frame_id = "base_footprint"
        self.pos.pose.position.x = 0
        self.pos.pose.position.y = 0
        self.pos.pose.position.z = 2
        self.pos.pose.orientation.x = 0
        self.pos.pose.orientation.y = 0
        self.pos.pose.orientation.z = 0
        self.pos.pose.orientation.w = 0
        # We need to send few setpoint messages, then activate OFFBOARD mode, to take effect
        k=0
        while k<10 and not rospy.is_shutdown():
            self.pos.header.stamp = rospy.Time.now()
            self.pub.publish(self.pos)
            rate.sleep()
            k = k + 1

    def navigate(self):
        rate = rospy.Rate(20)   # 10hz
        while not rospy.is_shutdown():
            rospy.loginfo("Navigating to (%d, %d, %d) [vl:%s] (%d, %d, %d)", 
                            self.pos.pose.position.x, self.pos.pose.position.y, self.pos.pose.position.z, 
                            self.violation,
                            self.local_position.pose.position.x, self.local_position.pose.position.y, self.local_position.pose.position.z)
            self.pos.header.stamp = rospy.Time.now()
            self.pub.publish(self.pos)
            try:
                rate.sleep()
            except rospy.ROSInterruptException:
                pass

    def is_at_position(self, x, y, z, offset):
        """offset: meters"""
        rospy.logdebug(
            "current position | x:{0:.2f}, y:{1:.2f}, z:{2:.2f}".format(
                self.local_position.pose.position.x, self.local_position.pose.
                position.y, self.local_position.pose.position.z))
        
        desired = np.array((x, y, z))
        pos = np.array((self.local_position.pose.position.x,
                        self.local_position.pose.position.y,
                        self.local_position.pose.position.z))
        return np.linalg.norm(desired - pos) < offset

    def reach_position(self,  x, y, z, w, timeout):
        
        # set a position setpoint
        self.pos.pose.position.x = x
        self.pos.pose.position.y = y
        self.pos.pose.position.z = z
        self.last_yaw_degrees = w

        yaw = math.radians(w)
        quaternion = quaternion_from_euler(0, 0, yaw)
        self.pos.pose.orientation = Quaternion(*quaternion)

        loop_freq = 20  # Hz
        rate = rospy.Rate(loop_freq)
        self.reached = False

        i=0
        while i  < (timeout * loop_freq):
            # wait until violation measures applied to return regular path
            if not self.violation: 
                if self.is_at_position(x,y,z,self.radius):
                    self.reached = True
                    rospy.loginfo("[%s] reached to (%d, %d, %d) ", self.reached ,self.pos.pose.position.x, self.pos.pose.position.y, self.pos.pose.position.z)
                    self.waypoints.pop()
                    break
                i+=1
                try:
                    rate.sleep()
                except rospy.ROSException as e:
                    break
            else:
                self.violation = False
                break
        

def demo_mission(offBoardMission):
    rate = rospy.Rate(20)
    start_time = time.time()
    recess_duration = 120 
    drone_rest = 5 

    offBoardMission.wait_for_topics(60)
    offBoardMission.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND,10, -1)

    offBoardMission.log_topic_vars()
    offBoardMission.send_dummy_setPoint()

    #offBoardMission.set_takeoff()
    offBoardMission.set_mode("OFFBOARD",  5)
    offBoardMission.set_arm(True, 5)
    #offBoardMission.wait_for_vtol_state()
      
    
    rospy.loginfo("run mission") 

    rospy.loginfo("Climb")
    offBoardMission.reach_position(0.0, 0.0, 15.0, 0, 5)
    '''
    rospy.loginfo("Sink")
    offBoardMission.reach_position(0.0, 00.0, 8.0, 0, 5)

    rospy.loginfo("Fly to the right")
    offBoardMission.reach_position(10.0, 10.0, 8.0, 0, 5)
    
    rospy.loginfo("Fly to the left")
    offBoardMission.reach_position(0.0, 0.0, 8.0, 0,  5)
    '''
    while (time.time()-start_time) < recess_duration:
        rospy.loginfo("Fly square")
        offBoardMission.waypoints.append([0.0, 0.0, 25.0, 90, 5])
        offBoardMission.waypoints.append([50.0, 0.0, 25.0, 180, 5])
        offBoardMission.waypoints.append([50.0, 50.0, 25.0, 270, 5])
        offBoardMission.waypoints.append([0.0, 50.0, 25.0, 0, 5])
      
        while offBoardMission.waypoints:
            offBoardMission.reach_position(*offBoardMission.waypoints[-1])  
            rate.sleep()
    ''' 
    offset_x = 0.0
    offset_y = 0.0
    offset_z = 10.0
    sides = 360
    radius = 20

    rospy.loginfo("Fly in a circle")
    offBoardMission.set(0.0, 0.0, 10.0, 3)   # Climb to the starting height first
    i = 0
    while not rospy.is_shutdown():
        x = radius * cos(i * 2 * pi / sides) + offset_x
        y = radius * sin(i * 2 * pi / sides) + offset_y
        z = offset_z

        wait = False
        delay = 0
        if (i == 0 or i == sides):
            # Let it reach the offBoardMission.
            wait = True
            delay = 5

        offBoardMission.set(x, y, z, delay, wait)

        i = i + 1
        rate.sleep()

        if (i > sides):
            rospy.loginfo("Fly home")
            offBoardMission.set(0.0, 0.0, 10.0, 5)
            break
 
    # Simulate a slow landing.
    offBoardMission.set(0.0, 0.0,  8.0, 5)
    offBoardMission.set(0.0, 0.0,  3.0, 5)
    offBoardMission.set(0.0, 0.0,  2.0, 2)
    offBoardMission.set(0.0, 0.0,  1.0, 2)
    offBoardMission.set(0.0, 0.0,  0.0, 2)
    offBoardMission.set(0.0, 0.0, -0.2, 2)
    '''
    offBoardMission.set_mode("AUTO.LAND", 5)

    offBoardMission.wait_for_landed_state(mavutil.mavlink.MAV_LANDED_STATE_ON_GROUND, 90, 0)
    
    offBoardMission.set_arm(False, 5)

    rospy.loginfo("Bye!")    

def initialize():
    rospy.init_node('OffBoardMission', log_level=rospy.INFO)
    mavros.set_namespace()  # initialize mavros module with default namespace
    offBoardMission = OffBoardMission()
    demo_mission(offBoardMission)

if __name__ == '__main__':
    try:
        initialize()
    except rospy.ROSInterruptException:
        pass

    
     