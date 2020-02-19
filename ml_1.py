# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 17:01:40 2019

@author: ali hussain
"""
f=open("D:/mystuff/patient.txt")
l=f.readline()
lines=f.readlines()

print(lines)

x=[]
y=[]
for line in lines:
    w=line.strip().split(",")
    ins=w[2:-1]
    inputs=[float(v) for v in ins]
    x.append(inputs)
    y.append(float(w[-1]))
    
print(x)
print(y)    

import numpy as np
ones=np.ones(6)
X=np.c_[ones,x]
print(X)

Y=np.c_[y]
print(Y)


def weight(X,Y):
    from numpy.linalg import inv
    left=inv(X.T.dot(X))
    rgt=X.T.dot(Y)
    return left.dot(rgt)

W=weight(X,Y)
print(W)


ycap=X.dot(W)
print(ycap)

print(np.c_[Y,ycap])










def accuracy(y,ycap,clossness):
    de=100-clossness
    dist=abs(y-ycap)/abs(y)*100
    pcnt=dist[dist<=de].size
    n=y.size
    return pcnt/n*100

accuracy(Y,ycap,90)





infile=open("D:/mystuff/newpat.txt")
hd=infile.readline()

data=infile.readlines()
print(data)


p=[]
for line in data:
    w=line.strip().split(",")
    ins=[float(v) for v in w[2:]]
    p.append(ins)
print(p)    

P=np.c_[np.ones(4),np.array(p)]
print(P)

chols=P.T.dot(W)
print(chols)

chols=np.array(chols)
print(chols)

c=np.ravel(chols)
print(c)

results=np.c_[P,c]
print(results)


out=open("D:/mystuff/newpats.txt","w")
header='"id","name","age","wgt","hgt","chol"'
out.write(header+ "\n")
for arr in results:
    line=arr[0].strip() ,",",arr[1]
    out.write(line + "\n")
out.close()

"""






