#!/usr/bin/env python3
# Basics ROS program to publish real-time streaming 
# video from your built-in webcam
# Author:
# - Addison Sears-Collins
# - https://automaticaddison.com
 
# Import the necessary libraries
import rospy # Python library for ROS
from sensor_msgs.msg import Image # Image is the message type
from offboard_camera.msg import ObjectDetection
from objectdetection.SocialDistance import SocialDistance
import cv2
from cv_bridge import CvBridge # Package to convert between ROS and OpenCV Images
import os

model_path = './model/yolov4-416.tflite'
label_path = './model/coco_labels.txt'

def publish_message():
  sd = SocialDistance(model_path, label_path)

  # Node is publishing to the video_frames topic using 
  # the message type Image
  pub = rospy.Publisher('navq/object_detection', ObjectDetection, queue_size=1)
     
  # Tells rospy the name of the node.
  # Anonymous = True makes sure the node has a unique name. Random
  # numbers are added to the end of the name.
  rospy.init_node('object_detection', anonymous=True)
     
  # Go through the loop 10 times per second
  rate = rospy.Rate(10) # 10hz
     
  # Create a VideoCapture object
  # The argument '0' gets the default webcam.
  cap = cv2.VideoCapture(0)
     
  # Used to convert between ROS and OpenCV images
  br = CvBridge()
 
  # While ROS is still running.
  while not rospy.is_shutdown():
     
      # Capture frame-by-frame
      # This method returns True/False as well
      # as the video frame.
      ret, frame = cap.read()
         
      if ret == True:
        # Print debugging information to the terminal
        rospy.loginfo('publishing video frame')
        
        sd.use_image(frame)
        annotated_image, number_of_violation = sd.process_image()     
        # Publish the image.
        # The 'cv2_to_imgmsg' method converts an OpenCV
        # image to a ROS image message
        data = ObjectDetection()
        data.image = br.cv2_to_imgmsg(annotated_image)
        data.violation = number_of_violation > 0
        pub.publish(data)
             
      # Sleep just enough to maintain the desired rate. note: detection process takes time , no need for sleep
      #rate.sleep()
         
if __name__ == '__main__':
  try:
    publish_message()
  except rospy.ROSInterruptException:
    pass
