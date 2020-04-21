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

        # Get Sensor Noise Covariance Matrix
        self.R = self.getSensorNoiseCovMat(self.deltaX, self.deltaY, self.deltaXdot, self.deltaYdot)

        # Get Initial Process Covariance Matrix
        self.Pk = self.getInitProcessCovMat(self.deltaPx, self.deltaPy, self.deltaPxdot, self.deltaPydot)

        # Get Process Noise Covariance Matrix
        #self.Q = self.getProcNoiseCovMat(self.noiseFactor)
        self.Q = np.eye(4)


    def getSensorNoiseCovMat(self, deltaX, deltaY, deltaXdot, deltaYdot):
        R = np.eye(4)
        R[0,0] = deltaX ** 2
        R[1,1] = deltaY ** 2
        R[2,2] = deltaXdot ** 2
        R[3,3] = deltaYdot ** 2

        return R


    def getInitProcessCovMat(self, deltaPx, deltaPy, deltaPxdot, deltaPydot):
        procCovMat = np.eye(4)
        procCovMat[0,0] = deltaPx ** 2
        procCovMat[1,1] = deltaPy ** 2
        procCovMat[2,2] = deltaPxdot ** 2
        procCovMat[3,3] = deltaPydot ** 2

        return procCovMat


    def getProcNoiseCovMat(self, noiseFactor):
        # continuos time model
        procNoiseCovMat =  np.eye(4)
        procNoiseCovMat[0,0] = (1/63)*math.pow(self.deltaT, 7)
        procNoiseCovMat[0,1] = (1/36)*math.pow(self.deltaT, 6)
        procNoiseCovMat[0,2] = (1/15)*math.pow(self.deltaT, 5)
        procNoiseCovMat[0,3] = (1/12)*math.pow(self.deltaT, 4)
        procNoiseCovMat[1,0] = (1/36)*math.pow(self.deltaT, 6)
        procNoiseCovMat[1,1] = (1/15)*math.pow(self.deltaT, 5)
        procNoiseCovMat[1,2] = (1/12)*math.pow(self.deltaT, 4)
        procNoiseCovMat[1,3] = (1/6)*math.pow(self.deltaT, 3)
        procNoiseCovMat[2,0] = (1/15)*math.pow(self.deltaT, 5)
        procNoiseCovMat[2,1] = (1/12)*math.pow(self.deltaT, 4)
        procNoiseCovMat[2,2] = (1/6)*math.pow(self.deltaT, 3)
        procNoiseCovMat[2,3] = (1/2)*math.pow(self.deltaT, 2)
        procNoiseCovMat[3,0] = (1/12)*math.pow(self.deltaT, 4)
        procNoiseCovMat[3,1] = (1/6)*math.pow(self.deltaT, 3)
        procNoiseCovMat[3,2] = (1/2)*math.pow(self.deltaT, 2)
        procNoiseCovMat[2,3] = (1/2)*math.pow(self.deltaT, 2)
        procNoiseCovMat[3,3] = self.deltaT

        return procNoiseCovMat


    def getInnovation(self, mesStateX, predictedStateX):
        return mesStateX - self.matrixH.dot(predictedStateX)

    def updateEstimationQ(self, mesStateX, predictedStateX ,kalmanGain):
        innovation = self.getInnovation(mesStateX, predictedStateX)
        self.Q = (self.alpha*self.Q) + (1-self.alpha)*(kalmanGain.dot(innovation).dot(innovation.T).dot(kalmanGain.T))


    def getResidual(self, mesStateX, kalmanStateX):
        return mesStateX - self.matrixH.dot(kalmanStateX)

    def updateEstimationR(self, mesStateX, kalmanStateX, procCovMat):
        residual = self.getResidual(mesStateX, kalmanStateX)
        self.R = (self.alpha*self.R) + (1-self.alpha)*(residual.dot(residual.T) + self.matrixH.dot(procCovMat).dot(self.matrixH.T))


    def getPredictedState(self, prevStateX):
        return (self.matrixA.dot(prevStateX) + (self.matrixB.dot(self.u)) + self.w)


    def predictProcCovarMat(self, prevCovarMat):
        procCovarMat = self.matrixA.dot(prevCovarMat).dot(self.matrixA.T) + self.Q
        # Set Non-Diagonal elements to zero
        for i in range(0, procCovarMat.shape[0]):
            for j in range(0, procCovarMat.shape[1]):
                if i != j:
                    procCovarMat[i, j] = 0
        return procCovarMat


    def getKalmanGain(self, procCovMat):
        num = procCovMat.dot(self.matrixH.T)
        den = self.matrixH.dot(procCovMat).dot(self.matrixH.T) + self.R
        kg = np.true_divide(num, den, where=(num != 0) & (den != 0))
        return kg


    def getObservedState(self, measuredState):
        return self.matrixC.dot(measuredState) + self.Z

    def getCorrectedState(self, predictedState, kg, obsState):
        return predictedState + kg.dot((obsState - self.matrixH.dot(predictedState)))

    def updateProcCovMat(self, kg, predictedProcCovMat):
        return (np.eye(4) - kg.dot(self.matrixH)).dot(self.Pkp)

    def getKalmanState(self, measuredStateMat):
        '''
        Kalman Filter originally suggests to Predict the state first,
        and then correct it based on measured/observed state. However,
        here I am predicting the next state in previous cycle and
        doing correction after getting the actual measurement in next
        cycle.

        For classic Kalman Filter the following line will be here instead
        of being at the end:

        ## Prediction ##
        self.stateXp = self.getPredictedState(self.stateX)
        self.Pkp = self.predictProcCovarMat(self.Pk)
        '''
        # Get observed state from the Measurement System ##
        self.obsState = self.getObservedState(measuredStateMat)

        # Compute kalman gain
        self.kg = self.getKalmanGain(self.Pkp)

        # Correction/Update
        self.stateX = self.getCorrectedState(self.stateXp, self.kg, self.obsState)
        ## Innovation Based Adaptive estimation of Q 
        # If the velocity mean is almost constant, then the best Q estimation will
        # be obtained by getProcNoiseCovarMat() whereas if the velocity mean keeps
        # increasing/decreasing with time, then the best Q estimation will be 
        # obtained by updateEstimationQ(). For vehicle tracking it will be best to 
        # use a constant Q as obtained by getProcNoiseCovarMat()
        self.updateEstimationQ(measuredStateMat, self.stateX, self.kg)
        # Residual based Adaptive Estimation of R
        self.updateEstimationR(measuredStateMat, self.stateX, self.Pkp)

        # Predict next state
        self.stateXp = self.getPredictedState(self.stateX)
        self.Pkp = self.getProcNoiseCovMat(self.Pk)

        print("Predicted State :\n", self.stateXp)
        print("\nPredicted Process Covariance Matrix :\n", self.Pkp)
        print("\nKalman Gain :\n", self.kg)
        print("\nObserved State :\n", self.obsState)
        print("\nKalman Corrected State :\n", self.stateX)
        print("\nProcess Covariance Matrix :\n", self.Pk)


        return [self.stateX, self.stateXp]

