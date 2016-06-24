#!/usr/bin/env python
"""
Getting Started Tutorial for RLPy
=================================

This file contains a very basic example of a RL experiment:
A simple Grid-World.
"""
__author__ = "Robert H. Klein"
from rlpy.Domains import GridWorld
from rlpy.Agents import Q_Learning
from rlpy.Representations import *
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import os
from DVRKVisualRL import *
from robot import *


def make_experiment(arm, exp_id=1, path="./Results/Tutorial/dvrk-planar"):
    """
    Each file specifying an experimental setup should contain a
    make_experiment function which returns an instance of the Experiment
    class with everything set up.

    @param id: number used to seed the random number generators
    @param path: output directory where logs and results are stored
    """
    opt = {}
    opt["exp_id"] = exp_id
    opt["path"] = path
    
    #u = [{'x': 0.0381381038389, 'y': 0.0348028884984}, {'x': 0.0553447503026, 'y': 0.0523395529395}]
    u = [{'x': 0.0333056007411, 'y': 0.0240999763421}]

    domain = DVRKPlanarDomain(arm, u[0])
    #domain = DVRKPlanarDomainHIRL(arm, u[0])
    opt["domain"] = domain

    # Representation
    representation = RBF(domain, num_rbfs=1000,resolution_max=25, resolution_min=25,
                         const_feature=False, normalize=False, seed=2)

    # Policy
    policy = eGreedy(representation, epsilon=0.1)

    # Agent
    opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=1.0,
                       learn_rate_decay_mode="boyan", boyan_N0=1000,
                       lambda_=0.0)
    opt["checks_per_policy"] = 1
    opt["max_steps"] = 400
    opt["num_policy_checks"] = 2
    experiment = Experiment(**opt)
    return experiment, domain, policy, representation

if __name__ == '__main__':
    
    arm = robot("PSM1")
    experiment, domain, policy, representation = make_experiment(arm, 1)
    experiment.run(visualize_steps=False,  # should each learning step be shown?
                   visualize_learning=False,  # show policy / value function?
                   visualize_performance=1)  # show performance runs?
    experiment.plot()
    experiment.save()
    domain.showExploration()

    #u = [{'x': 0.0366444748244, 'y': 0.0289956700025, 'z': -0.130718432791}, {'x': 0.0570558504403, 'y': 0.0556373344712, 'z': -0.120846617934}]
    
    """
    traj = []
    cur_state = domain.s0()[0]
    policy.turnOffExploration()
    for j in range(0, domain.episodeCap):
        traj.append(cur_state)
        terminal = domain.isTerminal()

        if terminal:
            break

        a = policy.pi(cur_state, terminal,domain.possibleActions())
        cur_state = domain.step(a)[1]

    print traj
    """

    #import pickle

    #pickle.dump({'representation': representation}, open('policy.p','wb'))