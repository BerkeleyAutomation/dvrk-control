#!/usr/bin/env python
from rlpy.Tools import plt, mpatches, fromAtoB
from rlpy.Domains.Domain import Domain
import numpy as np
import time
from copy import deepcopy
import tfx
import matplotlib.pyplot as plt

import rospy
from std_msgs.msg import String

import tfx
from geometry_msgs.msg import PointStamped, Point, PoseStamped, Pose
import pickle


"""
This is an example domain that uses rl to take the robot from a start
pose to a target pose in the plane
"""
class DVRKPlanarDomain(Domain):
    def __init__(self, arm, src):
        """
        :param traj takes a sequence of robot states
        """
        self.statespace_limits  = np.array([[0.012, 0.1], [0.02, 0.08], [-1,2]])
        self.episodeCap         = 50
        self.continuous_dims    = [0,1]
        self.DimNames           = ['X', 'Y']
        self.actions_num        = 4
        self.discount_factor    = 0.9
        self.stepsize = 0.002
        self.tolerance = 0.004
        self.scalefactor = 100
        self.visited_states = []

        self.feedback = None
        

        self.psm1 = arm
        self.src = src
        #elf.target = target
        self.time = 0
        
        self.z = -0.11867857918
        self.rot0 = [0.672831350856, 0.545250113857, 0.40137918567, 0.298152705764]

        #setup the ROS hooks
        rospy.Subscriber("/cutting/detected_line_feedback", String,
                         self.cache_feedback, queue_size=1)

        self.nextpos = rospy.Publisher("/cutting/next_position_cartesian", Pose)
 
        print "[DVRK Planar] Creating Object"
        super(DVRKPlanarDomain,self).__init__()

    def s0(self):
        self.home_robot()
        self.state = self.getCurrentRobotState()
        self.time = 0
        print "[DVRK Planar] Initializing and Homing DVRK", self.state
        return self.state, self.isTerminal(), self.possibleActions()

    def cache_feedback(self, msg):
        if rospy.is_shutdown():
            return

        self.feedback = str(msg)

    def getCurrentRobotState(self):
        pos = self.psm1.get_current_cartesian_position().position[:2]

        f = 2
        if self.feedback != None:
            if "left" in self.feedback:
                f = -1
            elif "right" in self.feedback:
                f = 1
            else:
                f = 0

        return np.array([pos[0,0], pos[1,0], f])

    def moveToPlanarPos(self, x, y):
        pos = [x,y,self.z]
        
        print "[DVRK Planar] Moving to", self.get_frame_psm1(pos, rot=self.rot0)

        frame = self.get_frame_psm1(pos,self.rot0)

        self.nextpos.publish(Pose(frame.position, frame.orientation))

        self.psm1.move_cartesian_frame_linear_interpolation(frame, speed=0.01)

        time.sleep(0.25)

    def cut(self, closed_angle=2.0, open_angle=80.0, close_time=1.5, open_time=1.35):
        """
        Closes and opens PSM1's grippers.
        """
        self.psm1.open_gripper(closed_angle)
        time.sleep(close_time)
        self.psm1.open_gripper(open_angle)
        time.sleep(open_time)

    def step(self,a):
        print  "[DVRK Planar] Action Applied", a, "at state=", self.state, "time=", self.time

        self.time = self.time + 1

        if a == 0:
            self.moveToPlanarPos(self.state[0]+self.stepsize, self.state[1])
            self.cut()
        elif a == 1:
            self.moveToPlanarPos(self.state[0]+self.stepsize, self.state[1]-self.stepsize)
            self.cut()
        elif a == 2:
            self.moveToPlanarPos(self.state[0]+self.stepsize, self.state[1]+self.stepsize)
            self.cut()


        s = self.getCurrentRobotState()

        self.state = np.copy(s)

        terminal = (a==3)



        reward = hirlReward(s,a)


        #print self.possibleActions(), s, np.array([self.target['x'], self.target['y']]), -np.linalg.norm(s-np.array([self.target['x'], self.target['y']]))

        #reward = -self.scalefactor*np.linalg.norm(s-np.array([self.target['x'], self.target['y']]))**2

        self.visited_states.append(np.copy(s))

        return reward, s, terminal, self.possibleActions()

    def isTerminal(self):
        return False
        #return (np.linalg.norm(self.state-np.array([self.target['x'], self.target['y']])) < self.tolerance)

    def possibleActions(self):
        rtn = []
        if self.state[0]+3*self.stepsize < self.statespace_limits[0,1]:
            rtn.append(0)

            if self.state[1]-self.stepsize > self.statespace_limits[1,0]:
                rtn.append(1) 

            if self.state[1]+self.stepsize < self.statespace_limits[1,1]:
                rtn.append(2)
        else:
            rtn.append(3)

        return rtn 

    def home_robot(self):
        self.moveToPlanarPos(self.src['x'], self.src['y'])

    def get_frame_psm1(self, pos, rot):
        """
        Gets a TFX pose from an input position/rotation for PSM1.
        """
        return tfx.pose(pos, rot)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        # import ipdb; ipdb.set_trace() 
        for k, v in self.__dict__.items():
            print k, v
            if k == "psm1" or k =='logger' or k == 'nextpos' or k== 'feedback':
                continue
            setattr(result, k, deepcopy(v, memo))
        result.psm1 = self.psm1
        result.logger = self.logger
        result.nextpos = self.nextpos
        result.feedback = self.feedback
        return result


    def showExploration(self):
        plt.figure()
        plt.scatter([i[0] for i in self.visited_states],[i[1] for i in self.visited_states], color='k')
        plt.scatter([self.src['x']],[self.src['y']], color='r')
        plt.scatter([self.target['x']],[self.target['y']], color='b')
        plt.show()

import pickle

"""
This is an example domain that uses rl to take the robot from a start
pose to a target pose in the plane
"""
class DVRKPlanarDomainHIRL(Domain):
    def __init__(self, arm, src):
        """
        :param traj takes a sequence of robot states
        """
        self.statespace_limits  = np.array([[0.012, 0.1], [0.02, 0.08], [0,1], [0,1], [0, 1]])
        self.episodeCap         = 20
        self.continuous_dims    = [0,1]
        self.DimNames           = ['X', 'Y']
        self.actions_num        = 7
        self.discount_factor    = 0.9
        self.stepsize = 0.002
        self.tolerance = 0.004
        self.scalefactor = 100
        self.started = False
        self.visited_states = []
        self.hirl = pickle.load(open("hirl.pkl","rb"))

        self.feedback = None
        

        self.psm1 = arm
        self.src = src
        #elf.target = target
        self.time = 0
        
        self.z = -0.1056567345189
        self.rot0 = [0.672831350856, 0.545250113857, 0.40137918567, 0.298152705764]

        #setup the ROS hooks
        rospy.Subscriber("/cutting/detected_line_feedback", String,
                         self.cache_feedback, queue_size=1)

        self.nextpos = rospy.Publisher("/cutting/next_position_cartesian", Pose)
 
        print "[DVRK Planar] Creating Object"
        super(DVRKPlanarDomainHIRL,self).__init__()

    def s0(self):
        self.home_robot()
        self.state = self.getCurrentRobotState()
        self.time = 0
        self.started = False
        print "[DVRK Planar] Initializing and Homing DVRK", self.state
        return self.state, self.isTerminal(), self.possibleActions()

    def cache_feedback(self, msg):
        if rospy.is_shutdown():
            return

        self.feedback = str(msg)

    def getCurrentRobotState(self):
        pos = self.psm1.get_current_cartesian_position().position[:2]


        if self.feedback != None:
            fl = 0
            fr = 0
            if "left" in self.feedback:
                fl = 1
            elif "right" in self.feedback:
                fr = 1
        else:
            fl = 1
            fr = 1

        #hard coded to one segment
        fis = 0
        if self.started:
            fis = 1

        return np.array([pos[0,0], pos[1,0], fl, fr, fis])

    def moveToPlanarPos(self, x, y):
        pos = [x,y,self.z]
        
        print "[DVRK Planar] Moving to", self.get_frame_psm1(pos, rot=self.rot0)

        frame = self.get_frame_psm1(pos,self.rot0)

        self.nextpos.publish(Pose(frame.position, frame.orientation))

        print "Current Position", pos, "Next Position", Pose(frame.position, frame.orientation)

        self.psm1.move_cartesian_frame_linear_interpolation(frame, speed=0.01)

        time.sleep(0.5)

    def cut(self, closed_angle=2.0, open_angle=80.0, close_time=1.5, open_time=1.35):
        """
        Closes and opens PSM1's grippers.
        """
        self.psm1.open_gripper(closed_angle)
        time.sleep(close_time)
        self.psm1.open_gripper(open_angle)
        time.sleep(open_time)

    def step(self,a):
        print  "[DVRK Planar] Action Applied", a, "at state=", self.state, "time=", self.time

        self.time = self.time + 1

        if a == 0:
            self.moveToPlanarPos(self.state[0]+self.stepsize, self.state[1])
            self.cut()
        elif a == 1:
            self.moveToPlanarPos(self.state[0]+self.stepsize, self.state[1]-self.stepsize)
            self.cut()
        elif a == 2:
            self.moveToPlanarPos(self.state[0]+self.stepsize, self.state[1]+self.stepsize)
            self.cut()
        elif a == 4:
            self.moveToPlanarPos(self.state[0]-3*self.stepsize, self.state[1])
        elif a == 5:
            self.moveToPlanarPos(self.state[0], self.state[1]-self.stepsize)
        elif a == 6:
            self.moveToPlanarPos(self.state[0], self.state[1]+self.stepsize)


        s = self.getCurrentRobotState()

        self.state = np.copy(s)

        terminal = (a==3)

        #hard coded to one segment
        if s[4] == 1:
            reward = np.dot(s,np.array(self.hirl["weights1"]))
        else:
            print "Not Started", self.state
            if np.sum(np.abs(self.state[0:2] - np.array(self.hirl["segment0"][0:2]))) < self.hirl["segment0tol"]:
                self.started = True

            reward = np.dot(s,np.array(self.hirl["weights0"]))


        print reward

        #print self.possibleActions(), s, np.array([self.target['x'], self.target['y']]), -np.linalg.norm(s-np.array([self.target['x'], self.target['y']]))

        #reward = -self.scalefactor*np.linalg.norm(s-np.array([self.target['x'], self.target['y']]))**2

        self.visited_states.append(np.copy(s))

        return reward, s, terminal, self.possibleActions()

    def isTerminal(self):
        if not self.started and self.time > 20:
            return True

        return False
        #return (np.linalg.norm(self.state-np.array([self.target['x'], self.target['y']])) < self.tolerance)

    def possibleActions(self):
        

        rtn = []

        if not self.started:
            if self.state[0] > 0.0238043900634:
                rtn.append(4)
            elif self.state[1]-self.stepsize > self.statespace_limits[1,0]:
                rtn.append(5)
            elif self.state[1]+self.stepsize < self.statespace_limits[1,1]:
                rtn.append(6)

            return rtn

        if self.state[0]+3*self.stepsize < self.statespace_limits[0,1]:
            rtn.append(0)

            if self.state[1]-self.stepsize > self.statespace_limits[1,0]:
                rtn.append(1) 

            if self.state[1]+self.stepsize < self.statespace_limits[1,1]:
                rtn.append(2)
        else:
            rtn.append(3)

        return rtn 

    def home_robot(self):
        x = 0.0238043900634 + np.squeeze(np.random.rand(1,1)*0.06)
        y = 0.0368345547882 + np.squeeze(min(np.random.randn(1,1)*0.0075, 0.01))
        #print "home", x, y
        self.moveToPlanarPos(x,y)
        #self.moveToPlanarPos(self.src['x'], self.src['y'])

    def get_frame_psm1(self, pos, rot):
        """
        Gets a TFX pose from an input position/rotation for PSM1.
        """
        return tfx.pose(pos, rot)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        # import ipdb; ipdb.set_trace() 
        for k, v in self.__dict__.items():
            print k, v
            if k == "psm1" or k =='logger' or k == 'nextpos' or k== 'feedback':
                continue
            setattr(result, k, deepcopy(v, memo))
        result.psm1 = self.psm1
        result.logger = self.logger
        result.nextpos = self.nextpos
        result.feedback = self.feedback
        return result


    def showExploration(self):
        plt.figure()
        plt.scatter([i[0] for i in self.visited_states],[i[1] for i in self.visited_states], color='k')
        plt.scatter([self.src['x']],[self.src['y']], color='r')
        plt.scatter([self.target['x']],[self.target['y']], color='b')
        plt.show()
