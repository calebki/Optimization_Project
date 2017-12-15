# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:52:15 2017

@author: Caleb
"""
import numpy as np
import numpy.linalg as LA

def tfocs(smoothF, gradF, nonsmoothF, projectorF, x0, 
          tol = 1e-8, gamma = 1e-4, method = 'AT'):
    """Returns optimal point and solution to optimization problem
    
    Keyword arguments:
        smoothF -- a smooth function
        gradF -- the gradient of the smooth function
        nonsmoothF -- a nonsmooth function
        projectorF -- prox function of the nonsmooth function
        x0 -- initial starting point
        tol -- tolerance threshold for convergence of optimization problem 
                (default 1e-8)
        gamma -- threshold for deciding how to backtrack using L (default 1e-5)
        method -- 'AT' for Auslender and Teboulle's method and 'LLM' for Lan, 
                  Lu, and Monteiro's method
    """
    alpha = 0.7
    beta = 0.5
    thetaNew = 1
    LNew = 1
    xNew = x0
    xBarNew = x0
    
    counter = 0
    
    while True:
        LOld = LNew
        LNew = LOld * alpha
        xOld = xNew
        xBarOld = xBarNew
        thetaOld = thetaNew
        counter = counter + 1
       
        while True:
            thetaNew = 2/(1 + np.sqrt(1 + 4*LNew / (np.power(thetaOld,2) * LOld)))
            y = (1 - thetaNew) * xOld + thetaNew * xBarOld
            xNew = projectorF(y - gradF(y)/LNew, 1/LNew)
            xBarNew = xNew
            if method == 'AT':
                xNew = (thetaNew)*xOld + (1-thetaNew)*xBarNew
            elif method == 'LLM':
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
            '''
            if smoothF(xNew) <= smoothF(y) + np.dot(gradF(y), xNew - y) + \
            0.5 * LNew * np.power(LA.norm(xNew - y),2):
                break
            LNew = LNew/beta
            '''
            #print("Function value: ", smoothF(xNew) + nonsmoothF(xNew))
            #print("Step size: ", LNew)
            #print("Difference between iterates: ", LA.norm(xNew - xOld)) 
            
            
        #print("Step size: ", LOld)
        #print("Function value: ", smoothF(xNew) + nonsmoothF(xNew))
        #print("Difference between iterates: ", LA.norm(xNew - xOld))    
        if LA.norm(xNew - xOld, 2)/max(1, LA.norm(xNew)) <= tol:
            break
        
        #if counter > 10:
            #break
    
    return np.array([xNew, smoothF(xNew) + nonsmoothF(xNew)])