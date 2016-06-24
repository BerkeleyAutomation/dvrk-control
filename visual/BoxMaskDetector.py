import rospy
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker
from std_msgs.msg import String
import cv2
import cv_bridge
import numpy as np
import scipy
import scipy.misc

import pickle
from matplotlib import pyplot as plt

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose
from tf_conversions import posemath
import image_geometry, tf
import tfx
from LineDetector import LineDetector
#from robot import *

class BoxMaskDetector:

    def __init__(self, offset=(0,0)):
        self.right_image = None
        self.left_image = None
        self.rcounter = 0
        self.lcounter = 0
        self.info = {'l': None, 'r': None}
        self.bridge = cv_bridge.CvBridge()
        self.block = False

        self.offset = offset

        self.LOWERB = np.array([0,0,0])
        self.UPPERB = np.array([80,80,80])

        self.robot_pose = None
        self.robot_next_pose = None

        self.camera_matrix = self.load_camera_matrix()

        rospy.init_node('image_saver', anonymous=True)

        rospy.Subscriber("/endoscope/left/image_rect_color", Image,
                         self.left_image_callback, queue_size=1)
        rospy.Subscriber("/dvrk/PSM1/position_cartesian_current", Pose,
                         self.position_callback, queue_size=1)
        rospy.Subscriber("/cutting/next_position_cartesian", Pose,
                         self.next_position_callback, queue_size=1)

        rospy.Subscriber("/endoscope/left/camera_info", CameraInfo,
                         self.camera_info_callback, queue_size=1)

        self.feedback = rospy.Publisher("/cutting/detected_line_feedback", String)

        rospy.spin()


    def left_image_callback(self, msg):
        if rospy.is_shutdown() or self.block or self.robot_pose == None or self.robot_next_pose == None:
            return

        self.left_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")

        c = LineDetector(self.left_image, show_plots=False)

        cur_position = self.get_pixel_from3D()
        next_position = self.get_pixel_from3D(position=self.robot_next_pose.position)

        print "Current Pose", self.robot_pose.position,"Next Pose", self.robot_next_pose.position

        feedback = c.query(cur_position, next_position)

        print "Current Pixel", cur_position,"Next Pixel", next_position, "feedback", feedback 

        if feedback == 1:
            self.feedback.publish("right")
        elif feedback == 0:
            self.feedback.publish("straight")
        elif feedback == -1:
            self.feedback.publish("left")
        else:
            self.feedback.publish("none")

        return

    def position_callback(self, msg):
        if rospy.is_shutdown() or self.block:
            return

        self.robot_pose = tfx.pose(msg)


    def next_position_callback(self, msg):
        if rospy.is_shutdown() or self.block:
            return

        self.robot_next_pose = tfx.pose(msg) 


    def camera_info_callback(self, msg):
        if rospy.is_shutdown() or self.block:
            return

        self.camera_info = msg 

    def load_camera_matrix(self):
        f = open("/home/davinci2/catkin_ws/src/endoscope_calibration/calibration_data/camera_matrix.p")
        info = pickle.load(f)
        f.close()
        return info

    def get_pixel_from3D(self, position=None):
        Trobot = np.zeros((4,4))
        Trobot[:3,:] = np.copy(self.camera_matrix)
        Trobot[3,3] = 1
        Rrobot = np.linalg.inv(Trobot)

        x = np.ones((4,1))

        if position == None:
            x[:3,0] = np.squeeze(self.robot_pose.position)
        else:
            x[:3,0] = np.squeeze(position)

        cam_frame = np.dot(Rrobot,x)

        Pcam = np.array(self.camera_info.P).reshape(3,4)

        V = np.dot(Pcam, cam_frame)

        V = np.array((int(V[0]/V[2]), int(V[1]/V[2])))

        V[0] = V[0] + self.offset[0]
        V[1] = V[1] + self.offset[1]

        return V

    def get_line_mask(self, position=None, tol=20):
        hsv = cv2.cvtColor(self.left_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWERB, self.UPPERB)
        self.left_mask = mask
        pix = self.get_pixel_from3D(position)

        if pix[0] < tol or pix[1] < tol:
            return False
        elif pix[0] > 1920-tol or pix[1] > 1080-tol:
            return False
        else:
            return (np.sum(mask[pix[1]-tol:pix[1]+tol, pix[0]-tol:pix[0]+ tol]) > 0)

    def get_line_distance(self, position=None, tol=50):
        hsv = cv2.cvtColor(self.left_image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.LOWERB, self.UPPERB)
        self.left_mask = mask
        pix = self.get_pixel_from3D(position)

        if pix[0] < tol or pix[1] < tol:
            return np.inf
        elif pix[0] > 1920-tol or pix[1] > 1080-tol:
            return np.inf
        else: 
            return self.closest_in_window(pix, mask, tol)[1]

    def closest_in_window(self, pix, mask, tol):
        closest=None
        dist = lambda u,v: np.abs(u[0]-v[0]) + np.abs(u[1]-v[1])

        for x in range(pix[1]-tol, pix[1]+tol):
            for y in range(pix[0]-tol, pix[0]+tol):
                if mask[x,y] != 0 and (closest == None or dist((y,x), pix) < dist(closest, pix)):
                    closest = (y,x)
        
        if closest == None:
            return None, np.inf
        else:
            print closest
            return closest, dist(closest, pix)


if __name__ == "__main__":


    a = BoxMaskDetector((-250,-89))
    pt = a.get_pixel_from3D()
    print pt
    img = np.copy(a.left_image)
    img[pt[1]-10:pt[1]+10,pt[0]-10:pt[0]+10, :] = [0,0,0]

    plt.figure()
    plt.imshow(img)
    plt.show()
    
    """
    pt = a.get_pixel_from3D()

    img = np.copy(a.left_image)
    print pt, a.get_line_distance()
    img[pt[1]-10:pt[1]+10,pt[0]-10:pt[0]+10, :] = [0,0,0]

    plt.figure()
    plt.imshow(a.left_mask)
    plt.show()

    plt.figure()
    plt.imshow(img)
    plt.show()

    plt.figure()
    """



