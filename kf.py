# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 18:08:22 2020

@author: Harshil
"""


import numpy as np
import math

# State Matrix
stateX = []

class KF_2D:
    def __init__(self, initStateX, delT):
        # Initial Conditions
        self.deltaT = delT
        self.x_2dot = 0 # Acceleration in x axis
        self.y_2dot = 0 # Acceleration in y axis

        # Observation Errors
        self.deltaX = 50
        self.deltaXdot = 5
        self.deltaY = 50
        self.deltaYdot = 5

        # Process Cov Matrix Errors
        self.deltaPx = 50
        self.deltaPxdot = 5
        self.deltaPy = 50
        self.deltaPydot = 5

        #            _     _
        #           |   x   |
        # State X = |   y   |
        #           | x_dot |
        #           | y_dot |
        #           |_     _|
        #

        # Initial state
        self.stateX = initStateX # state matrix
        self.obsState = 0 # observed Matrix
        self.u = np.array([[self.x_2dot], [self.y_2dot]]) # control matrix
        self.w = 0 # predicted state noise matrix
        self.noisefactor = 0.1 # Noise factor for process cov matrix
        self.alpha = 0.9 # Forgetting Factor to adaptively estimate R & Q
        self.Z = 0 # Measurement Noise
        self.stateXp = 0 # predicted state
        self.Pkp # predicted process cov matrix

        # Matrix A or F -> State Transition matrix
        #            _           _
        #           | 1  0  dt 0  |
        #       F = | 0  1  0  dt |
        #           | 0  0  1  0  |
        #           | 0  0  0  1  |
        #           |_           _|
        #
        self.matrixA = np.eye(4)
        self.matrixA[0, 2] = self.deltaT
        self.matrixA[1, 3] = self.deltaT


        # Matrix B or G -> Control Matrix
        # Matrix A or F -> State Transition matrix
        #            _                        _
        #           | 0.5 * dt^2       0       |
        #       G = |     0        0.5 * dt^2  |
        #           |     dt           0       |
        #           |_     0            dt    _|
        #

        self.matrixB = np.array([[(0.5*math.pow(self.deltaT, 2)), 0]
                                 [0, (0.5*math.pow(self.deltaT, 2))]
                                 [self.deltaT, 0],
                                 [0, self.deltaT]])

        # Matrix C or H -> Observation Matrix
        self.matrixC = np.eye(4)
        self.matrixH = np.eye(4)
