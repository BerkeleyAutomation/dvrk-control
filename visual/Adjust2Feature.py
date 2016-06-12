import rospy
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker
import cv2
import cv_bridge
import numpy as np
import scipy.misc
import pickle
from matplotlib import pyplot as plt

from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from robot import *

class Adjust2Feature:

    def __init__(self):
        self.right_image = None
        self.left_image = None
        self.rcounter = 0
        self.lcounter = 0
        self.info = {'l': None, 'r': None}
        self.bridge = cv_bridge.CvBridge()
        self.block = False


        self.camera_info = self.load_camera_info()
        self.r2c = self.load_inv_c2r_matrix()

        #========SUBSCRIBERS========#
        # image subscribers
        rospy.init_node('image_saver')
        rospy.Subscriber("/endoscope/left/image_rect_color", Image,
                         self.left_image_callback, queue_size=1)

        rospy.spin()


    def left_image_callback(self, msg):
        if rospy.is_shutdown() or self.block:
            return

        self.left_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        plt.imshow(self.left_image)
        plt.show()

    def load_camera_info(self):
        f = open("../../calibration_data/camera_left.p")
        info = pickle.load(f)
        f.close()
        return info

    def load_inv_c2r_matrix(self):
        f3 = open("../../calibration_data/camera_matrix.p", "rb")
        cmat = pickle.load(f3)
        f3.close()

        c2r = np.zeros((4,4))
        c2r[:3,:4] = cmat
        c2r[3,3] = 1
        return np.linalg.inv(c2r)

    def query_point(self, point=[0.0570198916239, 0.0178842822494, -0.128413618032]):
        
        robotPoint = np.ones((4,1))
        robotPoint[:3,0] = point

        camFrame = np.dot(self.r2c, robotPoint)

        Kcam = np.array(self.camera_info.K).reshape((3,3))
        K3Pix = np.dot(Kcam, camFrame[:3,0])

        return (int(K3Pix[0]/K3Pix[2]), int(K3Pix[1]/K3Pix[2]))

if __name__ == "__main__":
    a = Adjust2Feature()
    print a.query_point()

