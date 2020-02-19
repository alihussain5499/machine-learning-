# -*- coding: utf-8 -*-
"""
Created on Sun Dec  14 12:21:20 2019

@author: ali hussain
"""

f = open("D:/mystuff/patients2.txt")
h = f.readline()
lines = f.readlines()
print(len(lines))

x =[]
lbl=[]
for line in lines:
    w = line.strip().lower().split(",")
    ins = [float(v) for v in w[1:-1]]
    x.append(ins)
    lbl.append(w[-1])


import numpy as np
x=np.array(x)
print(x)

"""
Output:
[[25.   60.    5.9 ]
 [26.   67.    6.  ]
 [45.   78.    5.5 ]
 [56.   80.    5.9 ]
 [55.   81.    6.  ]
 [56.   79.    5.11]
 [55.   79.    5.4 ]
 [56.   89.    5.9 ]
 [49.   70.    5.6 ]
 [49.   75.    5.6 ]
 [52.   80.    5.7 ]]
"""

print(lbl)
"""
['high', 'moderate', 'high', 'low', 'low', 'moderate', 'high', 'high', 'moderate', 'high', 'high']
"""

# assigning unique numerical labels for 
# string labels.

ldict={word:i for i,word in enumerate(list(set(lbl)))}

print(ldict)
"""
{'moderate': 0, 'high': 1, 'low': 2}
"""

# transform string labels into numric labels
y = [ldict[word] for word in lbl]
print(y)
print(lbl)

"""
[1, 0, 1, 2, 2, 0, 1, 1, 0, 1, 1]
['high', 'moderate', 'high', 'low', 'low', 'moderate', 'high', 'high', 'moderate', ...]]

"""


def sd(x):
    num=((x-x.mean())**2).sum()
    den = x.size-1
    return (num/den)**0.5


def scale(x):
    return (x-x.mean())/sd(x)


def scaleMatrix(x):
    
    for col in range(x.shape[1]):
        x[:,col]= scale(x[:,col])
    ones = np.ones(x.shape[0])
    return np.c_[ones,x]

ins = np.array(x)
X = scaleMatrix(ins)
print(X)
"""

[[ 1.         -1.96389585 -2.05238632  0.73724967]
 [ 1.         -1.8771374  -1.16455628  1.09138706]
 [ 1.         -0.22872683  0.2306052  -0.67929991]
 [ 1.          0.72561614  0.48427093  0.73724967]
 [ 1.          0.63885769  0.61110379  1.09138706]
 [ 1.          0.72561614  0.35743807 -2.06043575]
 [ 1.          0.63885769  0.35743807 -1.0334373 ]
 [ 1.          0.72561614  1.62576669  0.73724967]
 [ 1.          0.11830698 -0.7840577  -0.32516252]
 [ 1.          0.11830698 -0.14989338 -0.32516252]
 [ 1.          0.37858233  0.48427093  0.02897488]]

"""

"""
 after scaling two problems will be solved. 
  1. weightage difference between the variables. 

  2. variance difference between the variables. 
  --> each variable variance will be approximated to 1. 

"""

print(sd(X[:,1]))
print(sd(X[:,2]))
print(sd(X[:,3]))

"""
0.9999999999999998
1.0000000000000002
1.0

"""
# num to binary
def NumToBinry(x):

    lbl=list(set(x))
    idx={l:i for i,l in enumerate(lbl)}
    y=[idx[l] for l in x]
    b=[]
    for v in y:
        barr=np.zeros(len(lbl))
        barr[v]=1
        b.append(barr)
    return b
    
Y=np.array(NumToBinry(y))

#initializing random weights. 

np.random.seed(101)
W=2*np.random.random((X.shape[1],Y.shape[1]))-1

print(W)

"""
[[ 0.03279726  0.14133517 -0.94305155]
 [-0.65695669  0.37055396  0.66779373]
 [-0.38606756  0.78722616  0.44308772]
 [-0.62012209  0.10845518 -0.29573609]]
"""

def model(x):
    return 1/(1+np.exp(-x))
def loss(y,ycap):
    return ((y-ycap)**2).mean()

# train the model.

ploss=0
flag=0
conv =0.00000001
for i in range(500000):
    ycap = model(X.dot(W))
    closs = loss(Y,ycap)
    e=Y-ycap
    if i%5000==0:
        print(" Loss ", closs)
    if abs(ploss-closs)<=conv:
        print("Training Completed ", i+1)
        flag=1
        break
    delta = e*(ycap*(1-ycap))
    gradients = X.T.dot(delta)
    W+= gradients
    ploss=closs
if flag==0:
    print("Training not completed ")

# Accuracy Testing. 

probs= model(X.dot(W))
print(probs)
"""
 predictions are probability values. 
[[3.19931389e-03 1.00000000e+00 1.25793105e-03]
 [4.39914743e-12 1.00000000e+00 1.31884134e-04]
 [3.92701622e-08 1.00000000e+00 1.59532768e-11]
 [1.44472929e-10 1.06442981e-10 9.95014258e-01]
 [1.32593150e-13 8.13239600e-02 9.99628403e-01]
 [9.95978636e-01 1.25084940e-01 6.64193705e-15]
 [6.62212741e-03 8.40637988e-01 3.62247231e-09]
 [8.56341559e-21 1.00000000e+00 4.22454009e-03]
 [9.94592302e-01 3.86423427e-02 3.28840530e-03]
 [3.82580070e-04 1.00000000e+00 8.36737348e-06]
 [9.65571711e-10 1.00000000e+00 9.68931866e-05]]
"""
"""
 Transform probabilites into binary array.
"""
probs[probs>0.5]=1
probs[probs<0.5]=0
print(probs)


"""
[[0. 1. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [0. 0. 1.]
 [0. 0. 1.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]
 [1. 0. 0.]
 [0. 1. 0.]
 [0. 1. 0.]]
"""

"""
 Convert binary array into numeric labels. 
"""

Yc = [int( np.where(arr==1)[0]) for arr in probs]

print(np.c_[y,Yc])
"""
[[1 1]
 [0 1]
 [1 1]
 [2 2]
 [2 2]
 [0 0]
 [1 1]
 [1 1]
 [0 0]
 [1 1]
 [1 1]]
"""
# function for accuracy test. 

def accuracy(y,ycap):
     c = y==ycap
     return  c[c==True].size/ y.shape[0]* 100

"""
    pcnt = number of predictions successful
    n= number of samples
    accuracy = pcnt/n * 100
"""

y = np.array(y)
# previously y is list.
accuracy(y,Yc)
#90.90

# Apply predictions on new data. 
"""
file : newpatients.txt

data:
"name","age","wgt","hgt"
aaa,28,67,6
bbbb,26,67,6.0
cccc,45,78,5.5
dddd,50,80,5.7
eeee,54,90,6.0
"""
# apply predictions on new data. 
new = open("D:/mystuff/newpatients.txt")
hd = new.readline()
data = new.readlines()
data = [line.strip() for line in data]
data = [line for line in data if line!='']
p=[]
for line in data:
    w = line.strip().split(",")
    ins = [float(v) for v in w[1:]]
    p.append(ins)
p=np.array(p)
print(p)

"""
[[28.  67.   6. ]
 [26.  67.   6. ]
 [45.  78.   5.5]
 [50.  80.   5.7]
 [54.  90.   6. ]]
"""

# scaling new data, with trained data mean and sd
for  i in range(p.shape[1]):
    p[:,i]= (p[:,i]-x[:,i].mean())/ sd(x[:,i])
print(p)

o = np.ones(p.shape[0])
P = np.c_[o,p]
print(P)
"""
[[ 1.         -1.7036205  -1.16455628  1.09138706]
 [ 1.         -1.8771374  -1.16455628  1.09138706]
 [ 1.         -0.22872683  0.2306052  -0.67929991]
 [ 1.          0.20506543  0.48427093  0.02897488]
 [ 1.          0.55209924  1.75259956  1.09138706]]
"""

# apply predictions. 
dstat = model(P.dot(W))
print(dstat)

dstat[dstat>0.5]=1
dstat[dstat<0.5]=0
print(dstat)


diabetic = [ int(np.where(v==1)[0]) for v in dstat]
print(diabetic)

numtostring = dict( zip(ldict.values(), ldict.keys()))
print(numtostring)


dbstat = [ numtostring[v] for v in diabetic]
print(dbstat)

head = '"name","age","wgt","hgt","dbstat"\n'
out = open("D:/mystuff/newreport.txt",'w')
out.write(head)
for  x,y in list(zip(data,dbstat)):
    line = x+','+y+'\n'
    out.write(line)
out.close()    

"""
output file :  newreport.txt

results:
"name","age","wgt","hgt","dbstat"
aaa,28,67,6,high
bbbb,26,67,6.0,high
cccc,45,78,5.5,high
dddd,50,80,5.7,high
eeee,54,90,6.0,high

"""









 








