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
from DVRKDomain import *
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
    u = [{'x': 0.0381381038389, 'y': 0.0348028884984, 'z': -0.122}, {'x': 0.0480933241889, 'y': 0.0586056886988, 'z': -0.129383575106}]

    domain = DVRK3DDomain(arm, u[0], u[1], sim=True)
    opt["domain"] = domain

    active_threshold = 0.01
    max_base_feat_sim = 0.5
    sparsify = 1

    # Representation
    kernel_width = (domain.statespace_limits[:, 1] - domain.statespace_limits[:, 0]) \
        / 10.0
    representation = KernelizediFDD(domain, sparsify=sparsify,
                               kernel=linf_triangle_kernel,
                               kernel_args=[kernel_width],
                               active_threshold=active_threshold,
                               discover_threshold=discover_threshold,
                               normalization=True,
                               max_active_base_feat=10,
                               max_base_feat_sim=max_base_feat_sim)

    # Policy
    policy = eGreedy(representation, epsilon=0.2)

    # Agent
    opt["agent"] = Q_Learning(representation=representation, policy=policy,
                       discount_factor=domain.discount_factor,
                       initial_learn_rate=0.975,
                       learn_rate_decay_mode="boyan", boyan_N0=1000,
                       lambda_=0.0)
    opt["checks_per_policy"] = 1
    opt["max_steps"] = 10000
    opt["num_policy_checks"] = 100
    experiment = Experiment(**opt)
    return experiment, domain

if __name__ == '__main__':
    
    arm = robot("PSM1")
    experiment, domain = make_experiment(arm, 1)
    experiment.run(visualize_steps=False,  # should each learning step be shown?
                   visualize_learning=False,  # show policy / value function?
                   visualize_performance=1)  # show performance runs?
    experiment.plot()
    experiment.save()
    domain.showExploration()