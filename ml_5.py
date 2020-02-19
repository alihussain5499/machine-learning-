# -*- coding: utf-8 -*-
"""
Created on Sun Dec  12 12:21:20 2019

@author: ali hussain
"""

"""
Binary Classification  with Gradient Descent algorithm.
---------------------------------
file: patients.txt

data:

"name","age","wgt","hgt","diabetic"
aaa,25,60,5.9,yes
bbbb,26,67,6.0,no
cccc,45,78,5.5,yes
dddd,56,80,5.9,no
eeee,55,81,6.0,no
ffffff,56,79,5.11,no
gggg,55,79,5.4,yes
iii,56,89,5.9,yes
hh,49,70,5.6,no
jjjj,49,75,5.6,yes
kkkkk,52,80,5.7,yes
"""
f = open("D:/mystuff/patients1.txt")
lines = f.readlines()[1:]
print(len(lines))
#11



x=[]
y=[]
for line in lines:
     w = line.strip().lower().split(",")
     ins =[float(v) for v in w[1:-1]]
     x.append(ins)
     if w[-1]=='yes':
        y.append(1)
     else:
        y.append(0)
print(x)
print("_"*40)
print(y)

"""
Output:

[[25.0, 60.0, 5.9], [26.0, 67.0, 6.0], [45.0, 78.0, 5.5], [56.0, 80.0, 5.9], [55.0, 81.0, 6.0], [56.0, 79.0, 5.11], [55.0, 79.0, 5.4], [56.0, 89.0, 5.9], [49.0, 70.0, 5.6], [49.0, 75.0, 5.6], [52.0, 80.0, 5.7]]
________________________________________
[1, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1]

"""

import numpy as np
x = np.array(x)
y = np.c_[y]
ins = np.array(x)

# functions for scaling. 
def sd(x):
    # x is a single column. 
    return  (((x-x.mean())**2).sum()/ (x.size-1))**0.5
def scale(x):
    # x is single column
    return (x-x.mean())/sd(x)
    
def scaleMatrix(x):
    ncol = x.shape[1]
    for i in range(ncol):
        col = x[:,i]
        x[:,i]=scale(col)
    o = np.ones(x.shape[0])
    return np.c_[o,x]


    
X = scaleMatrix(ins)
Y = y

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
print(Y)
# for classification models, dont scale,
# label matrix.(only features). 
"""
[[1]
 [0]
 [1]
 [0]
 [0]
 [0]
 [1]
 [1]
 [0]
 [1]
 [1]]

"""
# initiating random weights. 
np.random.seed(101)
W = 2*np.random.random((X.shape[1],Y.shape[1]))-1
print(W)
"""
[[ 0.03279726]
 [ 0.14133517]
 [-0.94305155]
 [-0.65695669]]
"""
# function for logistic
def model(z):
    #z is dot prodoct of inputs and weights
    return 1/(1+np.exp(-z))

# function for loss crossEntropy.
def loss(y,ycap):
    return   (y*np.log(ycap)).sum()*-1

def mse(y,ycap):
    return  ((y-ycap)**2).mean()

"""
 here we used two loss functions. 
 mse is just to monitor loss value at each
  iteration.  but derivative should be 
  applied on cross entropy.
"""

# function for derivative on cross Entropy

def derivative(output):
   return      output * ( 1- output)



# training model. 
ploss=0
flag=0
conv=1e-9
for i in range(4000000):
    ycap = model(X.dot(W))
    e = Y-ycap
    closs = mse(Y,ycap)
    if i%3000==0:
        print("Current loss at ", i+1 , " iterations ", closs)
    if abs(ploss-closs)<=conv:
         print("Training completed after ",i+1," iterations ")
         flag=1
         break
    
    delta = e * derivative(ycap) 
    W += X.T.dot(delta) * 0.06
    ploss=closs
if flag==0:
    print("training not completed")
    
    
"""
   if training not completed, run few more
 iterations till training completed
"""

"""
  after training do accuracy testing. 
   if satisifed, then apply predictions
  on new data. 
  Note: 
    we trained model with scaled data. 
    so new data should be scaled, with 
    mean and standard deviation of trained
    data 

  we will see these two phases in  Multinomial classfication example. 
"""

















