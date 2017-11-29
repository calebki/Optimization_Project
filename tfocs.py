# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:52:15 2017

@author: Caleb
"""
import numpy as np
import numpy.linalg as LA

def tfocs_AT(smoothF, affineF, projectorF, x0, tol = 1e-8):
    alpha = 0.9
    beta = 0.5
    thetaNew = 1
    z0 = 
    z1 = 
    LNew = LA.norm(smoothF(z0)[1] - smoothF(z1)[1], 2) / LA.norm(z0 - z1, 2)
    xNew = x0
    xBarNew = x0
    
    
    while True
        LOld = alpha * LNew
        xOld = xNew
        xBarOld = xBarNew
        thetaOld = thetaNew
        while True:
            y = (1 - thetaOld) * xOld + thetaOld * xBarOld
            xBarNew = 
            xNew = (1-theta)
            LHat =  
            LNew = 
            if LNew >= LHat:
                break
            LNew = max(L/beta, LHat)
            thetaNew = 2/(1+np.sqrt((1+4*LNew/(thetaOld^2 * LOld)))
        if stop(tol):
            break
        
def stop(tol):
    return LA.norm(xnew - xold, 2)/max(1, LA.norm(xnew)) <= tol 
    