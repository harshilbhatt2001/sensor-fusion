# -*- coding: utf-8 -*-
"""
Created on Fri Apr 17 01:21:22 2020

@author: Harshil
"""


import numpy as np


class KalmanFilter:


    def __init__(self, n):

        self.n = n
        self.I = np.eye(n)
        self.x = None
        self.P = None
        self.Q = None
        self.F = None

    def start(self, x, P, F, Q):

        self.x = x
        self.P = P
        self.F = F
        self.Q = Q
    
    def setQ(self, Q):
        self.Q = Q
    
    def updateF(self, dt):
        self.F[0,2], self.F[1,3] = dt, dt
    
    def getx(self)
        return self.x
    
    def predict(self):
        # x = F x
        # P = F P Ft + Q
        self.x = self.F.dot(self.x)
        self.P = self.F.dot(self.P).dot(self.F.T) + self.Q
    
    def update(self, z, H, Hx, R):
        # y = z - H x
        # S = H P Ht + R
        # K = P Ht S^-1
        # x = x + K y
        # P = (I - K H) * P
        y = z - Hx
        S = H.dot(P).dot(H.T) + R
        K = P.dot(H.T).dot(np.linalg.inv(S))

        self.x = self.x + K.dot(y)
        self.P = (self.I - K.dot(H)).dot(self.P)
    
    