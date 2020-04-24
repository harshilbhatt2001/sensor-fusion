# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 16:48:22 2020

@author: Harshil
"""


from kalmanfilter import KalmanFilter
from datapoint import DataPoint
from fusionekf import FusionEKF
from tools import get_RMSE
from helpers import parse_data, print_EKF_data, get_state_estimations
import numpy as np


lidar_R = np.array([[0.01, 0],
                    [0, 0.01]])

radar_R = np.array([[0.01, 0, 0],
                    [0, 0.000001, 0],
                    [0, 0, 0.01]])

lidar_H = np.array([[1, 0, 0, 0],
                    [0, 1, 0, 0]])

P = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1000, 0],
              [0, 0, 0, 1000]])

Q = np.zeros(4)
F = np.eye(4)

d = {
    'number_of_states': 4, 
    'initial_process_matrix': P,
    'radar_covariance_matrix': radar_R,
    'lidar_covariance_matrix': lidar_R, 
    'lidar_transition_matrix': lidar_H,
    'inital_state_transition_matrix': F,
    'initial_noise_matrix': Q, 
    'acceleration_noise_x': 5, 
    'acceleration_noise_y': 5
}

EKF1 = FusionEKF(d)
EKF2 = FusionEKF(d)


all_sensor_data, all_ground_truths = parse_data("data/data-1.txt")
all_state_estimations = get_state_estimations(EKF1, all_sensor_data)
px, py, vx, vy = get_RMSE(all_state_estimations, all_ground_truths)

print_EKF_data(all_sensor_data, all_ground_truths, all_state_estimations, 
               RMSE = [px, py, vx, vy])
