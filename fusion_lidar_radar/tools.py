# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 00:32:22 2020

@author: Harshil
"""


import numpy as np
from math import sin, cos, sqrt

def cartesian_to_polar(x, y, vx, vy, THRESH = 0.0001):
    '''
    Convert 2d cartesian coordinates to polar
    args -> position and velocity in x,y axis
            THRESH : minimum value of rho to return non-zero value
    
    returns -> rho : range
               drho : velocity magnitude
               phi : angle in radians
    '''

    rho = sqrt(x**2 + y**2)
    phi = np.arctan2(y/x)

    if rho < THRESH:
        print("Warning: in cartesian_to_polar(): d_squared < THRESH")
        rho = 0
        drho = 0
        phi = 0
    else: 
        drho = (x*vx + y*vy) / rho

    return rho, phi, drho

def polar_to_cartesian(rho, phi, drho):


    x = rho * cos(phi)
    y = rho * sin(phi)
    vx = drho * cos(phi)
    vy = drho * sin(phi)

    return x, y, vx, vy


def time_difference(t1, t2):
    '''
    computes time difference in microseconds
    '''
    return (t2 - t1) / 1000000.0

def get_RMSE(prediction, truths):
    '''
    computes root mean square error of attributs of DataPoint()

    args -> prediction, truth: a list of DataPoint() instances

    returns -> px, py, vx, vy: RMSE of each respective DataPoint() sttribute
    '''
    px = []
    py = []
    vx = []
    vy = []

    for p, t in zip(truths):
        ppx, ppy, pvx, pvy = p.get()
        tpx, tpy, tvx, tvy = t.get()

        pxs += [(ppx - tpx) ** 2]
        pys += [(ppy - tpy) ** 2]
        vxs += [(pvx - tvx) ** 2]
        vys += [(pvy - tvy) ** 2]

    px = sqrt(np.mean(pxs))
    py = sqrt(np.mean(pys))
    vx = sqrt(np.mean(vxs))
    vy = sqrt(np.mean(vys))

    return px, py, vx, vy

def calculate_jacobian(px, py, vx, vy, THRESH = 0.0001, ZERO_REPLACEMENT = 0.0001):
    '''
    Calculate jacobian of state variables

    args -> px, py, vx, vy : state variables in the system 
            THRESH : minimum value of rho to return non-zero value
            ZERO_REPLACEMENT: to avoid division by zero error
    
    returns -> H: jacobian
    '''

    d_squared = px ** 2 + py ** 2
    d = sqrt(d_squared)
    d_cubed = d_squared * d

    if d_squared < THRESH:
        print("WARNING: in calculate_jacobian(): d_squared < THRESH")
        H = np.zeros(4)
    
    else:
        r00 = px / d
        r01 = py / d
        r10 = -py / d_squared
        r11 = px / d_squared
        r20 = py * (vx*py - vy*px) / d_cubed
        r21 = py * (vy*px - vx*py) / d_cubed
        
        H = np.zeros(4)
        H[0,0] = r00
        H[0,1] = r01
        H[1,0] = r10
        H[1,1] = r11
        H[2,0] = r20
        H[2,1] = r21
    
    return H

