# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 12:21:20 2019

@author: ali hussain
"""

import numpy as np
x1=np.array([1,3,5,7,9,])
x2=np.array([2,4,6,8,10])
x3=np.array([0.1,0.2,0.3,0.4,0.5])
o=np.ones(len(x1))

L=np.c_[o,x1,x2,x3]
Q=np.c_[o,x1,x1**2,x2,x2**2,x3,x3**2]
C=np.c_[o,x1,x1**2,x1**3,x2,x2**2,x2**3,x3,x3**2,x3**3]
print(L)
print(Q)
print(C)

print(L.shape)
print(Q.shape)
print(C.shape)

def poly(x,d):
    M=[]
    for c in range(x.shape[1]):
        col=x[:,c]
        for i in range(d):
            M.append(col**(i+1))
    o=np.ones(x.shape[0])
    M=np.array(M).T
    return np.c_[o,M]

x=np.c_[x1,x2,x3]
print(x)

L=poly(x,1)
Q=poly(x,2)
C=poly(x,3)

print(L)
print(Q)
print(C)



def train(x,y,w,alpha,iter,conv=0.000000001):
    ploss=0
    flag=0
    for i in range(iter):
        ycap=predict(x,w)
        closs=loss(y,ycap)
        if i%1000==0:
            print("loss at iteration ",i,"is",closs)
        diff=abs(ploss-closs)
        if diff<=conv:
            print("Training completed after",i+1,"iteration")
            flag=1
            break
        grad=derivative(x,y,w)
        w+=grad*alpha
        ploss=closs
    if flag==0:
         print("Training not completed ")
    return w


def sd(x):
    return (((x-x.mean())**2).sum()/(x.size-1))**0.5
def scale(x):
    return (x-x.mean())/sd(x)
def scaleMatrix(x):
    for i in range(x.shape[1]):
        col=x[:,i]
        x[:,i]=scale(col)
    o=np.ones(x.shape[0])
    return np.c_[o,x]

def sigmoid(x):
    return 1/(1+np.exp(-x))
def loss(y,ycap):
    return ((y-ycap)**2).mean()
def derivative(x):
    return x*(1-x)

def predict(x,w):
    r=x
    for v in w:
        r=sigmoid(r.dot(v))
    return r


def accuracy(y,ycap,closeness):
    de=100-closeness
    dist=abs(y-ycap)/abs(y)*100
    pcnt=dist[dist<=de].size
    n=y.shape[0]
    return pcnt/n*100


XL=scaleMatrix(L)
XQ=scaleMatrix(Q)
XC=scaleMatrix(C)

print(XL)
print(XQ)
print(XC)
































