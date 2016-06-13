#!/usr/bin/env python
from rlpy.Tools import plt, mpatches, fromAtoB
from rlpy.Domains.Domain import Domain
import numpy as np
import time
from copy import deepcopy
import tfx
import matplotlib.pyplot as plt




"""
This is an example domain that uses rl to take the robot from a start
pose to a target pose in the plane
"""
class DVRKPlanarDomain(Domain):
    def __init__(self, arm, src, target):
        """
        :param traj takes a sequence of robot states
        """
        self.statespace_limits  = np.array([[0.025, 0.1], [0.02, 0.08]])
        self.episodeCap         = 30
        self.continuous_dims    = [0,1]
        self.DimNames           = ['X', 'Y']
        self.actions_num        = 4
        self.discount_factor    = 0.9
        self.stepsize = 0.002
        self.tolerance = 0.004
        self.scalefactor = 100
        self.visited_states = []
        

        self.psm1 = arm
        self.src = src
        self.target = target
        self.time = 0
        
        self.z = -0.122
        self.rot0 = [0.617571885272, 0.59489495214, 0.472153066551, 0.204392867261]
 
        print "[DVRK Planar] Creating Object"
        super(DVRKPlanarDomain,self).__init__()

    def s0(self):
        self.home_robot()
        self.state = self.getCurrentRobotState()
        self.time = 0
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

        time.sleep(1)

    def step(self,a):
        print  "[DVRK Planar] Action Applied", a, "at state=", self.state, "time=", self.time

        self.time = self.time + 1

        if a == 0:
            self.moveToPlanarPos(self.state[0]+self.stepsize, self.state[1])
        elif a == 1:
            self.moveToPlanarPos(self.state[0]-self.stepsize, self.state[1])
        elif a == 2:
            self.moveToPlanarPos(self.state[0], self.state[1]-self.stepsize)
        elif a == 3:
            self.moveToPlanarPos(self.state[0], self.state[1]+self.stepsize)

        """
        try:
            if a 
            #self.moveToPlanarPos(self.reference[self.time]['x'], self.reference[self.time]['y'])
        except IndexError:
            print "[DVRK Planar] Index error happened"
        """

        s = self.getCurrentRobotState()

        self.state = np.copy(s)

        terminal = self.isTerminal()

        print self.possibleActions(), s, np.array([self.target['x'], self.target['y']]), -np.linalg.norm(s-np.array([self.target['x'], self.target['y']]))

        reward = -self.scalefactor*np.linalg.norm(s-np.array([self.target['x'], self.target['y']]))**2

        self.visited_states.append(np.copy(s))

        return reward, s, terminal, self.possibleActions()

    def isTerminal(self):
        return (np.linalg.norm(self.state-np.array([self.target['x'], self.target['y']])) < self.tolerance)

    def possibleActions(self):
        rtn = []
        if self.state[0]+self.stepsize < self.statespace_limits[0,1]:
            rtn.append(0)

        if self.state[0]-self.stepsize > self.statespace_limits[0,0]:
            rtn.append(1) 

        if self.state[1]-self.stepsize > self.statespace_limits[1,0]:
            rtn.append(2) 

        if self.state[1]+self.stepsize < self.statespace_limits[1,1]:
            rtn.append(3)

        print self.statespace_limits[0,1]

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
            if k == "psm1" or k =='logger':
                continue
            setattr(result, k, deepcopy(v, memo))
        result.psm1 = self.psm1
        result.logger = self.logger
        return result


    def showExploration(self):
        plt.figure()
        plt.scatter([i[0] for i in self.visited_states],[i[1] for i in self.visited_states], color='k')
        plt.scatter([self.src['x']],[self.src['y']], color='r')
        plt.scatter([self.target['x']],[self.target['y']], color='b')
        plt.show()





"""
This is an example domain that uses rl to take the robot from a start
pose to a target pose in 3D
"""
class DVRK3DDomain(Domain):
    def __init__(self, arm, src, target, sim=False):
        """
        :param traj takes a sequence of robot states
        """
        self.statespace_limits  = np.array([[0.025, 0.1], [0.02, 0.08], [-0.13, -0.115]])
        self.episodeCap         = 100
        self.continuous_dims    = [0,1,2]
        self.DimNames           = ['X', 'Y', 'Z']
        self.actions_num        = 6
        self.discount_factor    = 0.9
        self.stepsize = 0.002
        self.tolerance = 0.004
        self.scalefactor = 100
        self.simCap = 100000
        self.sim = sim
        self.visited_states = []
        

        self.psm1 = arm
        self.src = src
        self.target = target
        self.time = 0
        
        self.z = -0.122
        self.rot0 = [0.617571885272, 0.59489495214, 0.472153066551, 0.204392867261]
 
        print "[DVRK 3D] Creating Object"
        super(DVRK3DDomain,self).__init__()

    def s0(self):
        self.home_robot()
        self.state = self.getCurrentRobotState()
        self.time = 0
        print "[DVRK 3D] Initializing and Homing DVRK", self.state, self.possibleActions()

        return self.state, self.isTerminal(), self.possibleActions()

    def getCurrentRobotState(self):
        pos = self.psm1.get_current_cartesian_position().position[:3]
        return np.array([pos[0,0], pos[1,0], pos[2,0]])

    def moveTo3DPos(self, x, y, z):
        pos = [x,y,z]
        
        print "[DVRK 3D] Moving to", self.get_frame_psm1(pos, rot=self.rot0)

        if not self.sim:
            self.psm1.move_cartesian_frame_linear_interpolation(self.get_frame_psm1(pos,self.rot0), speed=0.01)
            time.sleep(0.2)
            return self.getCurrentRobotState()
        else:
            #print np.array([x+np.random.randn(1,1)*0.001,y+np.random.randn(1,1)*0.001,z+np.random.randn(1,1)*0.001])
            return np.array([  np.squeeze(x+np.random.randn(1,1)*0.0005) , np.squeeze(y+np.random.randn(1,1)*0.0005) , np.squeeze(z+np.random.randn(1,1)*0.0005) ])

    def step(self,a):
        print  "[DVRK 3D] Action Applied", a, "at state=", self.state, "time=", self.time

        self.time = self.time + 1

        if self.sim and len(self.visited_states) > self.simCap:
            print "!!!!!!!!!!!![DVRK 3D] Beginning Real Executions!!!!!!!!!"
            self.showExploration()
            time.sleep(2)
            self.sim = False

        if a == 0:
            s = self.moveTo3DPos(self.state[0]+self.stepsize, self.state[1], self.state[2])
        elif a == 1:
            s = self.moveTo3DPos(self.state[0]-self.stepsize, self.state[1], self.state[2])
        elif a == 2:
            s = self.moveTo3DPos(self.state[0], self.state[1]-self.stepsize, self.state[2])
        elif a == 3:
            s = self.moveTo3DPos(self.state[0], self.state[1]+self.stepsize, self.state[2])
        elif a == 4:
            s = self.moveTo3DPos(self.state[0], self.state[1], self.state[2]-self.stepsize)
        else:
            s = self.moveTo3DPos(self.state[0], self.state[1], self.state[2]+self.stepsize)

        

        self.state = np.copy(s)

        terminal = self.isTerminal()

        print self.possibleActions(), s, np.array([self.target['x'], self.target['y'], self.target['z']])
        print -np.linalg.norm(s-np.array([self.target['x'], self.target['y'], self.target['z']]))

        reward = -self.scalefactor*np.linalg.norm(s-np.array([self.target['x'], self.target['y'], self.target['z']]))**2

        self.visited_states.append(np.copy(s))

        return reward, s, terminal, self.possibleActions()

    def isTerminal(self):
        return (np.linalg.norm(self.state-np.array([self.target['x'], self.target['y'], self.target['z']])) < self.tolerance)

    def possibleActions(self):
        rtn = []
        if self.state[0]+self.stepsize < self.statespace_limits[0,1]:
            rtn.append(0)

        if self.state[0]-self.stepsize > self.statespace_limits[0,0]:
            rtn.append(1) 

        if self.state[1]-self.stepsize > self.statespace_limits[1,0]:
            rtn.append(2) 

        if self.state[1]+self.stepsize < self.statespace_limits[1,1]:
            rtn.append(3)

        if self.state[2]-self.stepsize > self.statespace_limits[2,0]:
            rtn.append(4) 

        if self.state[2]+self.stepsize < self.statespace_limits[2,1]:
            rtn.append(5)

        #print self.statespace_limits[0,1]

        return rtn 

    def home_robot(self):
        self.moveTo3DPos(self.src['x'], self.src['y'], self.src['z'])

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


    def showExploration(self):
        plt.figure()
        plt.scatter([i[0] for i in self.visited_states],[i[1] for i in self.visited_states], color='k')
        plt.scatter([self.src['x']],[self.src['y']], color='r')
        plt.scatter([self.target['x']],[self.target['y']], color='b')
        plt.show()
