#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 20:55:09 2017

@author: Sanjana
"""


import numpy as np
import numpy.linalg as LA

data = np.genfromtxt('diabetes.csv', delimiter=',')
#X = np.array(data[:, :-1], dtype='float')
#Y = np.array(data[:, -1], dtype='int')
n, p = 4000, 100
k = 5
X = np.random.normal(size = (n, p))
beta = np.append(np.zeros(p - k), np.full(k, 10))
error = np.random.normal(scale = 0.01, size = n)
#X=np.hstack((np.ones((np.shape(X)[0],1)),X))
Y = X.dot(beta) + error

s=0
#s = LA.norm(np.dot(X.T,error), ord = np.inf)

def smoothF(b):
    return (LA.norm(Y-np.dot(X,b)))/2.0
def gradF(b):
    return np.dot(np.transpose(X),np.dot(X,b))-np.dot(np.transpose(X),Y)
def nonsmoothF(b):
    return s*LA.norm(b,1)
def projectorF(b,t):
    return np.where(b<-s*t,b+s*t,np.where(abs(b)<=s*t,0,b-s*t))

a,b = tfocs(smoothF, gradF, nonsmoothF, projectorF,np.zeros(p), 
          tol = 1e-8, gamma = 1e-4, method = 'AT')


'''
def SoftThreshold(x,y):
    return np.where(x<-y,x+y,np.where(abs(x)<=y,0,x-y))
def prox(x,t):
    return SoftThreshold(x,s*t)
def projectorF(b,s):
    return prox(b,s)
''' 