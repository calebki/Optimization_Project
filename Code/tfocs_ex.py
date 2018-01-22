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
m = [500, 100]
npratio = [2, 0.5]
pkratio = [1.25]
settings = np.empty(shape = (len(m)*len(npratio)*len(pkratio), 4), dtype = object)
help = 0
for n in m:
    for p in npratio:
        for q in pkratio:
            settings[help] = [n, int(n*p), int(n*(p/q)), 1]
            help = help + 1

numsim = 100

counter = 0

for n, p, k, sigma in settings:
            
        for i in range(0,numsim):
            X = np.random.normal(size = (n, p))
            beta = np.append(np.zeros(p - k), np.full(k, 10))
            error = np.random.normal(scale = sigma, size = n)
            Y = X.dot(beta) + error

            lambdas = [0, 
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
                    
                t0 = time.time()
                bstarAT, pstarAT, countAT = tfocs(smoothF, gradF, nonsmoothF, projectorF, 
                                                  np.zeros(p, dtype = np.longdouble), 
                                                  tol = 1e-4, gamma = 1e-1, solver = 'AT')
                t1 = time.time()
                t2 = time.time()
                bstarLLM, pstarLLM, countLLM = tfocs(smoothF, gradF, nonsmoothF, projectorF, 
                                                     np.zeros(p, dtype = np.longdouble), 
                                                     tol = 1e-4, gamma = 1e-1, solver = 'LLM')
                t3= time.time()
                df.loc[counter] = [n, p, k, sigma, s, 'AT', t1 - t0, 
                      LA.norm(beta - bstarAT) / LA.norm(beta), countAT]
                df.loc[counter + 1] = [n, p, k, sigma, s, 'LLM', t3 - t2, 
                      LA.norm(beta - bstarLLM) / LA.norm(beta), countLLM]
                counter = counter + 2

df.to_csv("simdata2.csv")