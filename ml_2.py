# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 11:48:29 2019

@author: ali hussain
"""

import numpy as np
x=np.array([1,2,3,4,5,6])
y=np.array([2,3.9,6.1,7.93,9.98,11.25])

a=0
b=0

def predict(x,a,b):
    ycap=a+b*x
    return ycap

def loss(y,ycap):
    return ((y-ycap)**2).mean()

def grad_a(y,ycap):
    return (y-ycap).mean()

def grad_b(x,y,ycap):
    return (x*(y-ycap)).mean()

# Trial - 1
    
ycap=predict(x,a,b)
print(ycap)

loss(y,ycap)

da=grad_a(y,ycap)

db=grad_b(x,y,ycap)

a+=da*0.02
b+=db*0.02

def train(x,y,w,alpha,iter,conv=0.00000001):
    ploss=0
    flag=0
    a=w[0]
    b=w[1]
    for i in range(iter):
        ycap=predict(x,a,b)
        closs=loss(y,ycap)
        diff=abs(ploss-closs)
        if diff<=conv:
            print("Training completed after ",i+1,"iteration")
            flag=1
            break
        
        if i%200==0:
            print("loss at ",i+1,"iteration",closs)
        da=grad_a(y,ycap)
        db=grad_b(x,y,ycap)
        a+=da*alpha
        b+=db*alpha
        ploss=closs
    if flag==0:
        print("Training not completed , run more iteration ")
    return(a,b)

a=0
b=0    
w=train(x,y,[a,b],0.02,100)    


w=train(x,y,w,0.02,200)

w=train(x,y,w,0.02,1000)

w=train(x,y,w,0.02,10000,0.0000000001)




























