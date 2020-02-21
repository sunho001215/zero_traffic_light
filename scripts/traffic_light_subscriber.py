#!/usr/bin/env python
import rospy
import cv2
import sys
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
#sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')

def callback_Image(data):
    br = CvBridge()
    cv_image = br.imgmsg_to_cv2(data)
    cv2.imshow("Traffic_Light_Image", cv_image)
    cv2.waitKey(10)

def callback_String(data):
    print(data)

def traffic_light_subscriber():
    rospy.init_node("traffic_light_subscriber_node", anonymous = True)
    rospy.Subscriber("/traffic_light_Image", Image, callback_Image)
    rospy.Subscriber("/traffic_light_String", String, callback_String)
    rospy.spin()

if __name__ == "__main__":
    traffic_light_subscriber()
