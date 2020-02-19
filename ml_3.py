# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 10:36:09 2019

@author: ali hussain
"""

f=open("D:/mystuff/patient.txt")
header=f.readline()
lines=f.readlines()
print(lines)

x=[]
y=[]
for line in lines:
    w=line.strip().split(",")
    ins=[float(v) for v in w[2:-1]]
    x.append(ins)
    y.append(float(w[-1]))

print(x)
print(y)    

import numpy as np
x=np.array(x)
ones=np.ones(x.shape[0])
X=np.c_[ones,x]
Y=np.c_[y]

print(X)
print(Y)


np.random.seed(101)
W=2*np.random.random((X.shape[1],1))-1
print(W.shape)
print(W)



def predict(x,w):
    return x.dot(w)

def loss(y,ycap):
    return ((y-ycap)**2).mean()

def derivative(x,y,w):
    ycap=predict(x,w)
    return (x.T.dot(y-ycap))/x.shape[0]


ycap1=predict(X,W)
print(loss(Y,ycap1))


grad=derivative(X,Y,W)
W+=grad*0.02

ycap2=predict(X,W)
print(loss(Y,ycap2))

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
    

ins=np.array(x)
out=np.array(Y)
X=scaleMatrix(ins)
YL=scaleMatrix(out)

print(x)

print(ins)

print(X)

print(YL)

YL=out
print(YL)

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

theta=np.array(W)
theta=train(X,YL,theta,0.02,100)

theta=train(X,YL,theta,0.02,10000)

theta=train(X,YL,theta,0.02,100000)
    

def accuracy(y,ycap,closeness):
    de=100-closeness
    dist=abs(y-ycap)/abs(y)*100
    pcnt=dist[dist<=de].size
    n=y.shape[0]
    return pcnt/n*100

ycap=predict(X,theta)
print(ycap)

yc=ycap*sd(Y)+Y.mean()
print(yc)

accuracy(Y,yc,90)


np.c_[Y,yc]

new=open("D:/mystuff/newpat.txt")
hd=new.readline()
file=new.readlines()
print(file)
p=[]
for line in file:
    w=line.strip().split(",")
    ins=[float(v) for v in w[2:]]
    p.append(ins)
    
p=np.array(p)    
print(p)    


for i in range(p.shape[1]):
    s=sd(x[:,i])
    m=x[:,i].mean()
    p[:,i]=(p[:,i]-m)/s


print(p)


o=np.ones(len(file))
P=np.c_[o,p]
pcap=predict(P,theta)
print(pcap)

chols=pcap*sd(Y) + Y.mean()
chols=chols.ravel()
chols

res=[d.strip()+ ","+ str(c) for d,c in list(zip(file,chols)) ]
res

outfile=open("D:/mystuff/cholres.txt",'w')
hdr='"id","name","age","wgt","hgt","chols"\n'
outfile.write(hdr)
for line in res:
    outfile.write(line+"\n")
outfile.close()    












