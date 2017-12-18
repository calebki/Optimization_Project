#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:55:09 2017

@author: Sanjana
"""
from tfocs import tfocs
import numpy as np
import numpy.linalg as LA
import time
import pandas as pd

df = pd.DataFrame(columns = ['n', 'p', 'k', 'sigma', 'lambda', 
                        'method', 'time', 'error', 'numiters'])
settings = np.array([[500, 100, 10, 1], [50, 100, 10, 1], 
                     [500, 100, 10, 0.1], [50, 100, 10, 0.1]], dtype = object)
numsim = 1

counter = 0

for n, p, k, sigma in settings:
    
    X = np.random.normal(size = (n, p))
    beta = np.append(np.zeros(p - k), np.full(k, 10))
    error = np.random.normal(scale = sigma, size = n)
    Y = X.dot(beta) + error

    lambdas = [0, LA.norm(np.dot(X.T,error), ord = np.inf)/2, 
         LA.norm(np.dot(X.T,error), ord = np.inf) + 0.001]

    for s in lambdas:
        def smoothF(b):
            return (LA.norm(Y-np.dot(X,b)))/2.0
        def gradF(b):
            return np.dot(np.transpose(X),np.dot(X,b))-np.dot(np.transpose(X),Y)
        def nonsmoothF(b):
            return s*LA.norm(b,1)
        def projectorF(b,t):
            return np.where(b < -s * t, b + s * t, 
                            np.where(abs(b) <= s * t, 0, b - s * t))
            
        for i in range(0,numsim):
            
            t0 = time.time()
            bstarAT, pstarAT, countAT = tfocs(smoothF, gradF, nonsmoothF, projectorF, 
                                              np.zeros(p, dtype = np.longdouble), 
                                              tol = 1e-8, gamma = 1, solver = 'AT')
            t1 = time.time()
            bstarLLM, pstarLLM, countLLM = tfocs(smoothF, gradF, nonsmoothF, projectorF, 
                                                 np.zeros(p, dtype = np.longdouble), 
                                                 tol = 1e-8, gamma = 1, solver = 'LLM')
            t2= time.time()
            df.loc[counter] = [n, p, k, sigma, s, 'AT', t1 - t0, 
                  LA.norm(beta - bstarAT), countAT]
            df.loc[counter + 1] = [n, p, k, sigma, s, 'LLM', t2 - t1, 
                  LA.norm(beta - bstarLLM), countLLM]
            counter = counter + 2

df.to_csv("simdata.csv")