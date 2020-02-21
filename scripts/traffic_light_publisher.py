#!/usr/bin/env python
# license removed for brevity

from ctypes import *

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import math
import random
import os
import sys
from cv_bridge import CvBridge
#sys.path.remove('/opt/ros/melodic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import time
import darknet



def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def cvDrawBoxes(detections, img):
    for detection in detections:
        x, y, w, h = detection[2][0],\
            detection[2][1],\
            detection[2][2],\
            detection[2][3]
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 1)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img

netMain = None
metaMain = None
altNames = None

def traffic_light_publish():
    ###################################
    pub = rospy.Publisher('/traffic_light_Image', Image, queue_size = 10)
    pub2 = rospy.Publisher('/traffic_light_String', String, queue_size = 10)
    rospy.init_node('traffic_light_publisher_node', anonymous = True)
    rate = rospy.Rate(10)
    ###################################
    global metaMain, netMain, altNames
    configPath = "/home/sunho/catkin_ws/src/traffic_light/scripts/yolo-obj.cfg"
    weightPath = "/home/sunho/catkin_ws/src/traffic_light/scripts/yolo-obj_last.weights"
    metaPath = "/home/sunho/catkin_ws/src/traffic_light/scripts/obj.data"
    ###################################
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass    
    ################################### 
    cap = cv2.VideoCapture(0)
    cap.set(3, 1280)
    cap.set(4, 720)
    print("Starting the YOLO loop...")
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    while not rospy.is_shutdown():
        prev_time = time.time()
        ret, frame_read = cap.read()
        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=0.25)
        max_area = 0
        max_idx = -1
        for idx, detection in enumerate(detections):
            if detection[2][2]*detection[2][3] > max_area:
                max_idx = idx
        if max_idx != -1:
            pub2.publish(detections[max_idx][0])
#            print(detections[max_idx][0])
        image = cvDrawBoxes(detections, frame_resized)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#        print(1/(time.time()-prev_time))
#        cv2.imshow('Demo', image)
        br = CvBridge()
        pub.publish(br.cv2_to_imgmsg(image))
        cv2.waitKey(3)
        rate.sleep()
    cap.release()
    out.release()

if __name__ == "__main__":
    try:
        traffic_light_publish()
    except rospy.ROSInterruptException:
        pass


