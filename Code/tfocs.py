# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:52:15 2017

@author: Caleb
"""
import numpy as np
import numpy.linalg as LA

def tfocs(smoothF, gradF, nonsmoothF, projectorF, x0, 
          tol = 1e-8, gamma = 1e-2, solver = 'AT'):
    """Returns optimal point and solution to optimization problem using 
       Auslender and Teboulle's method
    
    Keyword arguments:
        smoothF -- a smooth function
        gradF -- the gradient of the smooth function
        nonsmoothF -- a nonsmooth function
        projectorF -- prox function of the nonsmooth function
        x0 -- initial starting point
        tol -- tolerance threshold for convergence of optimization problem 
                (default 1e-8)
        gamma -- threshold for deciding how to backtrack using L (default 1e-5)
    """
    alpha = 0.9
    beta = 0.5
    thetaNew = 1
    LNew = 1
    xNew = x0
    xBarNew = x0
    itercount = 0
    
    while True:
        itercount = itercount + 1
        xOld = xNew
        xBarOld = xBarNew
        
        LOld = LNew
        LNew = LOld * alpha
        thetaOld = thetaNew
       
        while True:
            thetaNew = 2/(1 + np.sqrt(1 + 4*(LNew/LOld) / (np.power(thetaOld,2))))
            y = (1 - thetaNew) * xOld + thetaNew * xBarOld
            xBarNew = projectorF(xBarOld - gradF(y)/(LNew * thetaNew), 
                                  1/(LNew * thetaNew))
            
            if solver == 'AT':
                xNew = (1 - thetaNew) * xOld +  thetaNew * xBarNew
            else:
                xNew = projectorF(y - gradF(y)/LNew, 1/LNew)
            
            if smoothF(y) - smoothF(xNew) >= gamma * smoothF(xNew):                        
                LHat = \
                2*(smoothF(xNew) - smoothF(y) - (gradF(y).dot(xNew - y))) \
                /np.power(LA.norm(xNew - y),2) 
            else:
                LHat = 2 * np.absolute(
                        np.dot(y - xNew, gradF(xNew) - gradF(y))) \
                        /np.power(LA.norm(xNew - y),2)
            if LNew >= LHat:
                break
            LNew = max(LNew/beta, LHat)
            

        if LA.norm(xNew - xOld, 2)/max(1, LA.norm(xNew)) <= tol:
            break
    
    if solver == 'AT':
        xNew = projectorF(xNew - gradF(xNew)/LNew, 1/LNew)
   
    return np.array([xNew, smoothF(xNew) + nonsmoothF(xNew), itercount])