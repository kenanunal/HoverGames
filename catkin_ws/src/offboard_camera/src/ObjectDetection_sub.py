#!/usr/bin/env python3
# Description:
# - Subscribes to real-time streaming video from your built-in webcam.
#
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
 
# Import the necessary libraries
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from offboard_camera.msg import ObjectDetection
import cv2
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
 
i = 1

def callback(data):
  global i 
  # Used to convert between ROS and OpenCV images
  br = CvBridge()
 
  # Convert ROS Image message to OpenCV image
  current_frame = br.imgmsg_to_cv2(data.image)

  # Display image
  #cv2.imshow("camera", current_frame)
  #check /home/navq/.ros
  imgfile = 'frame'+str(i)+'.jpeg' 
  cv2.imwrite(imgfile, current_frame)
  i += 1 

  rospy.loginfo("received video frame %d - violation %s", i, data.violation) 

      
def receive_message():
 
  # Tells rospy the name of the node.
  # Anonymous = True makes sure the node has a unique name. Random
  # numbers are added to the end of the name. 
  rospy.init_node('video_sub_py', anonymous=True)
   
  # Node is subscribing to the video_frames topic
  rospy.Subscriber('navq/object_detection', ObjectDetection, callback)
 
  # spin() simply keeps python from exiting until this node is stopped
  rospy.spin()
 
  # Close down the video stream when done
  cv2.destroyAllWindows()
  
if __name__ == '__main__':
  receive_message()
