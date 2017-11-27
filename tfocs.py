# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 20:52:15 2017

@author: Caleb
"""
import numpy as np
import numpy.linalg as LA

'''
Implementing Algorithm 9 in TFOCS.pdf
The smooth function g is represented as g(x) = gbar(Astar(z)) + <b,z>
'''
def AT_Solver(g, A, b, zOld, LHat, alpha = 0.9, beta, k):
    zBarOld = z
    AStar = A.conj().T#set as conjugate transpose
    zAOld = 
    zBarAOld = 
    gConj = 
    thetaOld = float("inf")
    theta0 = 1
    LOld = LHat

    for i in k:
        LNew = alpha * LOld
        while True
            thetaNew = 2/(1+(1+4*))
            yNew = (1 - thetaNew) * zOld + thetaNew * zBarOld
            yANew = (1 - thetaNew) * zAOld + thetanew * zBarAOld
            gConjNew = 
            gNew = 
            zBarNew =
            zNew = 
            LHat = 
            if LNew >= LHat:
                break
            LNew = max(LNew/beta, LHat)
        
    