#!/usr/bin/env python
from rlpy.Tools import plt, mpatches, fromAtoB
from rlpy.Domains.Domain import Domain
import numpy as np
import time
from copy import deepcopy
import tfx

class DVRKPlanarDomain(Domain):
    def __init__(self, arm, u):
        """
        :param traj takes a sequence of robot states
        """
        self.statespace_limits  = np.array([[0.025, 0.1], [0.02, 0.08]])
        self.episodeCap         = len(u) - 1
        self.continuous_dims    = [0,1]
        self.DimNames           = ['X', 'Y']
        self.actions_num        = 1
        self.discount_factor    = 0.9
        

        self.psm1 = arm
        self.reference = u
        self.time = 0
        
        self.z = -0.122
        self.rot0 = [0.617571885272, 0.59489495214, 0.472153066551, 0.204392867261]
 
        print "[DVRK Planar] Creating Object"
        super(DVRKPlanarDomain,self).__init__()

    def s0(self):
        self.home_robot()
        self.state = self.getCurrentRobotState()
        print "[DVRK Planar] Initializing and Homing DVRK", self.state
        return self.state, self.isTerminal(), self.possibleActions()

    def getCurrentRobotState(self):
        pos = self.psm1.get_current_cartesian_position().position[:2]
        return np.array([pos[0,0], pos[1,0]])

    def moveToPlanarPos(self, x, y, fake=False):
        pos = [x,y,self.z]
        
        print "[DVRK Planar] Moving to", self.get_frame_psm1(pos, rot=self.rot0)

        if not fake:
            self.psm1.move_cartesian_frame_linear_interpolation(self.get_frame_psm1(pos,self.rot0), speed=0.01)

        time.sleep(2)

    def step(self,a):
        print  "[DVRK Planar] Action Applied", a, "at state=", self.state, "time=", self.time

        self.time = self.time + 1

        try:
            self.moveToPlanarPos(self.reference[self.time]['x'], self.reference[self.time]['y'])
        except IndexError:
            print "[DVRK Planar] Index error happened"

        s = self.getCurrentRobotState()

        terminal = self.isTerminal()

        return 0, s, terminal, self.possibleActions()

    def home_robot(self):
        self.moveToPlanarPos(self.reference[0]['x'], self.reference[0]['y'])

    def isTerminal(self):
        if self.time == len(self.reference):
            return True
        else:
            return False 

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
            if k == "psm1" or k =='logger':
                continue
            setattr(result, k, deepcopy(v, memo))
        result.psm1 = self.psm1
        result.logger = self.logger
        return result




"""
This is an example domain in which the dvrk "phantom" graps
a point along a trajectory
"""

class DVRKPhantomGraspDomain(Domain):
    def __init__(self, arm, u):
        """
        :param traj takes a sequence of robot states
        """
        self.statespace_limits  = np.array([[0.025, 0.1], [0.02, 0.08]])
        self.episodeCap         = len(u) - 1
        self.continuous_dims    = [0,1]
        self.DimNames           = ['X', 'Y']
        self.actions_num        = 2
        self.discount_factor    = 0.9
        

        self.psm1 = arm
        self.reference = u
        self.time = 0
        
        self.z = -0.122
        self.rot0 = [0.617571885272, 0.59489495214, 0.472153066551, 0.204392867261]
 
        print "[DVRK Phantom Grasp] Creating Object"
        super(DVRKPhantomGraspDomain,self).__init__()

    def s0(self):
        self.home_robot()
        self.state = self.getCurrentRobotState()
        self.time = 0
        print "[DVRK Phantom Grasp] Initializing and Homing DVRK", self.state
        return self.state, self.isTerminal(), self.possibleActions()

    def getCurrentRobotState(self):
        pos = self.psm1.get_current_cartesian_position().position[:2]
        return np.array([pos[0,0], pos[1,0]])

    def moveToPlanarPos(self, x, y, fake=False):
        pos = [x,y,self.z]
        
        print "[DVRK Phantom Grasp] Moving to", self.get_frame_psm1(pos, rot=self.rot0)

        if not fake:
            self.psm1.move_cartesian_frame_linear_interpolation(self.get_frame_psm1(pos,self.rot0), speed=0.01)

        time.sleep(2)

    def cut(self, closed_angle=2.0, open_angle=80.0, close_time=2.5, open_time=2.35):
        """
        Closes and opens PSM1's grippers.
        """
        self.psm1.open_gripper(closed_angle)
        time.sleep(close_time)
        self.psm1.open_gripper(open_angle)
        time.sleep(open_time)

    def step(self,a):
        print  "[DVRK Phantom Grasp] Action Applied", a, "at state=", self.state, "time=", self.time

        self.time = self.time + 1

        r = 0

        """
        Must only cut at certain points
        """
        if a == 1 and self.time == 2:
            r = 1
        elif a == 1 and self.time != 2:
            r = -1

        try:
            self.moveToPlanarPos(self.reference[self.time]['x'], self.reference[self.time]['y'])

            if a == 1:
                print "Cutting"
                self.cut()


        except IndexError:
            print "[DVRK Phantom Grasp] Index error happened"

        s = self.getCurrentRobotState()

        terminal = self.isTerminal()

        return r, s, terminal, self.possibleActions()

    def home_robot(self):
        self.moveToPlanarPos(self.reference[0]['x'], self.reference[0]['y'])
        self.psm1.open_gripper(80.0)

    def isTerminal(self):
        if self.time == len(self.reference):
            return True
        else:
            return False 

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
            if k == "psm1" or k =='logger':
                continue
            setattr(result, k, deepcopy(v, memo))
        result.psm1 = self.psm1
        result.logger = self.logger
        return result


