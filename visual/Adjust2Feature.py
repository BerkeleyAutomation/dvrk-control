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
from geometry_msgs.msg import Pose
from tf_conversions import posemath
import image_geometry, tf
import tfx
#from robot import *

class Adjust2Feature:

    def __init__(self):
        self.right_image = None
        self.left_image = None
        self.rcounter = 0
        self.lcounter = 0
        self.info = {'l': None, 'r': None}
        self.bridge = cv_bridge.CvBridge()
        self.block = False


        #self.camera_info = self.load_camera_info()
        self.camera_matrix = self.load_camera_matrix()
        #self.camera_model = image_geometry.PinholeCameraModel()
        #self.camera_model.fromCameraInfo(self.camera_info)

        rospy.init_node('image_saver', anonymous=True)

        rospy.Subscriber("/endoscope/left/image_rect_color", Image,
                         self.left_image_callback, queue_size=1)
        rospy.Subscriber("/dvrk/PSM1/position_cartesian_current", Pose,
                         self.position_callback, queue_size=1)
        rospy.Subscriber("/endoscope/left/camera_info", CameraInfo,
                         self.camera_info_callback, queue_size=1)

        rospy.spin()
        #self.tfl = tf.TransformListener()

        #========SUBSCRIBERS========#
        # image subscribers

    def left_image_callback(self, msg):
        if rospy.is_shutdown() or self.block:
            return

        self.left_image = self.bridge.imgmsg_to_cv2(msg, "rgb8")
        #plt.imshow(self.left_image)
        #plt.show()

    def position_callback(self, msg):
        if rospy.is_shutdown() or self.block:
            return

        self.robot_pose = tfx.pose(msg) 
        

        #print self.robot_pose 
        #plt.imshow(self.left_image)
        #plt.show()

    def camera_info_callback(self, msg):
        if rospy.is_shutdown() or self.block:
            return

        self.camera_info = msg 

    def load_camera_matrix(self):
        f = open("/home/davinci2/catkin_ws/src/endoscope_calibration/calibration_data/camera_matrix.p")
        info = pickle.load(f)
        f.close()
        return info

    def query_point(self):
        #ps = self.robot_pose.msg.PoseStamped()
        #print ps
        #self.tfl.waitForTransform('/endoscope_frame','/world', rospy.Time.now(), rospy.Duration(0.5))
        #print a
        #point = self.tfl.transformPoint(self.camera_model.tf_frame, ps)
        Trobot = np.zeros((4,4))
        Trobot[:3,:] = np.copy(self.camera_matrix)
        Trobot[3,3] = 1
        Rrobot = np.linalg.inv(Trobot)

        print self.camera_info        
        #print 'Rrobot', Rrobot, self.camera_info.K

        x = np.ones((4,1))
        x[:3,0] = np.squeeze(self.robot_pose.position)

        #print 'x', x

        cam_frame = np.dot(Rrobot,x)

        #print 'cam_frame', cam_frame

        Pcam = np.array(self.camera_info.P).reshape(3,4)

        #print 'p_cam', Pcam
                
        V = np.dot(Pcam, cam_frame)
        #print V

        #print np.shape(np.array(self.camera_info.D))
        #Dcam = np.array(self.camera_info.D).reshape(2,2)
        V = np.array((int(V[0]/V[2]), int(V[1]/V[2])))

        #print "res", V
        return V

    def loopChessBoardPoints(self):
        pts = pickle.load(open('/home/davinci2/catkin_ws/src/endoscope_calibration/calibration_data/endoscope_points.p', 'rb'))
        for p in pts:
            Rrobot = np.zeros((4,4))
            Rrobot[:3,:] = np.copy(self.camera_matrix)
            Rrobot[3,3] = 1

            cmat = np.copy(self.camera_matrix)
            x = np.ones((4,1))
            x[:3,0] = np.squeeze(p)

            xr = np.ones((4,1))
            xr[:3,0] = np.squeeze(np.dot(cmat, x))

            print p, xr, np.dot(np.linalg.inv(Rrobot), xr)







if __name__ == "__main__":


    a = Adjust2Feature()
    pt = a.query_point()

    a.loopChessBoardPoints()


    img = np.copy(a.left_image)
    print pt
    img[pt[1]-10:pt[1]+10,pt[0]-10:pt[0]+10, :] = [0,0,0]

    plt.imshow(img)
    plt.show()

