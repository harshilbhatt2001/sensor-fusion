# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 01:34:22 2020

@author: Harshil
"""


from kalmanfilter import KalmanFilter
from datapoint import DataPoint
from tools import calculate_jacobian, cartesian_to_polar, time_difference
import numpy as np


class FusionEKF:


    def __init__(self, d):
        self.initialized = False
        self.timestamp = 0
        self.n = d['number of states']
        self.P = ['initial_process_matrix']
        self.F = d['inital_state_transition_matrix']
        self.Q = d['initial_noise_matrix']
        self.radar_R = d['radar_covariance_matrix']
        self.lidar_R = d['lidar_covariance_matrix']
        self.lidar_H = d['lidar_transition_matrix']
        self.a = (d['acceleration_noise_x'], d['acceleration_noise_y'])
        self.kalmanFilter = KalmanFilter(self.n)
    
    def updateQ(self, dt):
        
        dt2 = dt ** 2
        dt3 = dt ** 3
        dt4 = dt ** 4

        x,y = self.a

        Q = np.zeros(4)
        
        Q[0,0] = dt4 * x / 4
        Q[0,2] = dt3 * x / 2
        Q[1,1] = dt4 * y / 4
        Q[1,3] = dt3 * y / 2
        Q[2,0] = dt3 * x / 2
        Q[2,2] = dt2 * x
        Q[3,1] = dt3 * y / 2
        Q[3,3] = dt2 * y
        
        self.kalmanFilter.setQ(Q)
    
    def update(self, data):

        dt = time_difference(self.timestamp, data.get_timestamp())
        self.timestamp = data.get_timestamp()

        self.kalmanFilter.updateF(dt)
        self.updateQ(dt)
        self.kalmanFilter.predict()

        z = np.matrix(data.get_raw()).T
        x = self.kalmanFilter.getx()

        if data.get_name == 'radar':
            px = x[0,0]
            py = x[1,0]
            vx = x[2,0]
            vy = x[3,0]
            rho, phi, drho = cartesian_to_polar(px, py, vx, vy)
            H = calculate_jacobian(px, py, vx, vy)
            Hx = (np.array([rho, phi, drho]).reshape((3,1))).T
            R = self.radar_R
        
        elif data.get_name == 'lidar':

            H = self.lidar_H
            Hx = self.lidar_H * x
            R = self.lidar_R
        
        self.kalmanFilter.update(z, H, Hx, R)

    def start(self, data):

        self.timestamp = data.get_timestamp()
        x = np.matrix([data.get()]).T
        self.kalmanFilter.start(x, self.P, self.F, self.Q)
        self.initialized = True
    
    def process(self, data):

        if self.initialized:
            self.update(data)
        else:
            self.start(data)
    
    def get(self):
        return self.kalmanFilter.getx()
            
