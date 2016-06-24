import rospy, pickle, time
from robot import *
import numpy as np
import PyKDL
import multiprocessing
import tfx
from geometry_msgs.msg import PointStamped, Point, PoseStamped, Pose
#import fitplane
from scipy.interpolate import interp1d
#from shape_tracer import plot_points
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

"""
This file contains utilities that are used for a trajectory following curve cutting model.
"""


class ClosedLoopCut(object):

    def __init__(self, factor = 4, fname="../../calibration_data/gauze_pts.p", simulate=False):
        self.psm1 = robot("PSM1")
        self.psm2 = robot("PSM2")
        self.pts = self.interpolation(self.load_robot_points(fname), factor)
        self.factor = factor
        self.nextpos = rospy.Publisher("/cutting/next_position_cartesian", Pose)
        self.simulate = simulate
        self.traj = []

    """
    Method homes the robot to a given start position
    """
    def home_robot(self):
        pos = [0.0333056007411, 0.0440999763421, -0.11067857918]#-0.0485527311586]
        rot = [0.672831350856, 0.545250113857, 0.40137918567, 0.298152705764]

        if not self.simulate:
            self.psm1.move_cartesian_frame(self.get_frame_psm1(pos,rot))
            time.sleep(10)
        else:
            print "[ClosedLoopCut] Simulated Move to", self.get_frame_psm1(pos,rot)

    """
    Any initialization scripts go here
    """

    def initialize(self):
        """
        Initialize both arms to a fixed starting position/rotation.
        """
        self.home_robot()

        print "[Closed Loop Cutting] Initializing and Homing the Robot"
        
        return

    def get_frame_psm1(self, pos, rot):
        """
         Gets a TFX pose from an input position/rotation for PSM1.
        """
        return tfx.pose(pos, rot)

    def cut(self, closed_angle=2.0, open_angle=80.0, close_time=2.5, open_time=2.35):
        """
        Closes and opens PSM1's grippers.
        """
        if not self.simulate:
            self.psm1.open_gripper(closed_angle)
            time.sleep(close_time)
            self.psm1.open_gripper(open_angle)
            time.sleep(open_time)
        else:
            print "[ClosedLoopCut] Simulated Cut"

    def load_robot_points(self, fname):
        lst = []
        f3 = open(fname, "rb")
        while True:
            try:
                pos2 = pickle.load(f3)
                lst.append(pos2)
            except EOFError:
                f3.close()
            except ValueError:
                break;
        return np.matrix(lst)

    def interpolation(self, arr, factor):
        """
        Given a matrix of x,y,z coordinates, output a linearly interpolated matrix of coordinates with factor * arr.shape[1] points.
        """
        x = arr[:, 0]
        y = arr[:, 1]
        z = arr[:, 2]
        t = np.linspace(0,x.shape[0],num=x.shape[0])
        to_expand = [x, y, z]
        for i in range(len(to_expand)):
            print t.shape, np.ravel(to_expand[i]).shape
            spl = interp1d(t, np.ravel(to_expand[i]))
            to_expand[i] = spl(np.linspace(0,len(t), len(t)*factor))
        new_matrix = np.matrix(np.r_[0:len(t):1.0/factor])
        for i in to_expand:
            new_matrix = np.concatenate((new_matrix, np.matrix(i)), axis = 0)
        return new_matrix.T[:,1:]

    def get_frame_next(self, pos, nextpos, offset=0.003, angle=None):
        """
        Given two x,y,z coordinates, output a TFX pose that points the grippers to roughly the next position, at pos.
        """
        if angle:
            angle = angle
        else:
            angle = get_angle(pos, nextpos)
        print angle
        pos[2] -= offset
        # pos[0] += offset/3.0
        rotation = [94.299363207+angle, -4.72728031036, 86.1958002688]
        rot = tfx.tb_angles(rotation[0], rotation[1], rotation[2])
        frame = tfx.pose(pos, rot)
        return frame

    def get_angle(self, pos, nextpos):
        """
        Returns angle to nextpos in degrees
        """
        delta = nextpos - pos
        theta = np.arctan(delta[1]/delta[0]) * 180 / np.pi
        if delta[0] < 0:
            return theta + 180
        return theta

    def calculate_xy_error(self, desired_pos):
        actual_pos = np.ravel(np.array(psm1.get_current_cartesian_position().position))[:2]
        return np.linalg.norm(actual_pos - desired_pos)

    def doTask(self):
        self.initialize()
        N = self.pts.shape[0]
        
        angles = []
        for i in range(N-1):
            pos = self.pts[i,:]
            nextpos = self.pts[i+1,:]
            angle = self.get_angle(np.ravel(pos), np.ravel(nextpos))
            angles.append(angle)

        for i in range(len(angles)-2):
            angles[i] = 0.5 * angles[i] + 0.35 * angles[i+1] + 0.15 * angles[i+2]
            angles = savgol_filter(angles, self.factor * (N/12) + 1, 2)



        for i in range(N-1):
            self.cut()
            pos = self.pts[i,:]
            nextpos = self.pts[i+1,:]
            frame = self.get_frame_next(np.ravel(pos), np.ravel(nextpos), offset=0.004, angle = angles[i])
            self.nextpos.publish(Pose(frame.position, frame.orientation))

            if not self.simulate:
                self.psm1.move_cartesian_frame(frame)
            else:
                print "[ClosedLoopCut] Simulated Move to", frame
                time.sleep(1)

            curpt = np.ravel(np.array(self.psm1.get_current_cartesian_position().position))
            self.pts[i,:] = curpt
            self.pts[i+1,:2] = savgol_filter(self.pts[:,:2], 5, 2, axis=0)[i+1,:]
            self.traj.append(self.psm1.get_current_cartesian_position())
        

if __name__ == "__main__":


    c = ClosedLoopCut(simulate=False)
    #c.initialize()

    c.doTask()

    import pickle
    pickle.dump(c.traj, open("traj-real.p","wb"))
