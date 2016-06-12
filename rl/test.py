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
from rlpy.Representations import Tabular
from rlpy.Policies import eGreedy
from rlpy.Experiments import Experiment
import os
from DVRKDomain import *
from robot import *


def make_experiment(arm, exp_id=1, path="./Results/Tutorial/gridworld-qlearning"):
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
    
    u = [{'x': 0.0381381038389, 'y': 0.0348028884984}, {'x': 0.0510237465354, 'y': 0.0400495340115}, {'x': 0.0727044526597, 'y': 0.058846162056}]
    domain = DVRKPhantomGraspDomain(arm, u)
    opt["domain"] = domain

    # Representation
    representation = Tabular(domain, discretization=20)

    # Policy
    policy = eGreedy(representation, epsilon=0.5)

    # Agent
    opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.1,
                       learn_rate_decay_mode="boyan", boyan_N0=100,
                       lambda_=0.)
    opt["checks_per_policy"] = 1
    opt["max_steps"] = 10
    opt["num_policy_checks"] = 5
    experiment = Experiment(**opt)
    return experiment

if __name__ == '__main__':
    
    arm = robot("PSM1")
    experiment = make_experiment(arm, 1)
    experiment.run(visualize_steps=False,  # should each learning step be shown?
                   visualize_learning=False,  # show policy / value function?
                   visualize_performance=1)  # show performance runs?
    experiment.plot()
    experiment.save()