import numpy as np
from numpy import dot
from numpy.linalg import inv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D


#import math
from math import exp

#helper functinon to to get class probability
def findClassPro(X_,beta_):
    disfun = dot(X_,beta_)
    p_ = []
    for i in range(disfun.shape[0]):
        pi = 1/(exp(-disfun[i])+1)
        p_.append(pi)
    print(p_)
    return p_

#helper function to create 2Dgrid of test points
def grid2d(start, end, num=50):
    """Create an 2D array where each row is a 2D coordinate"""
    dom = np.linspace(start, end, num)
    X0, X1 = np.meshgrid(dom, dom)
    return np.column_stack([X0.flatten(), X1.flatten()])




#load the data
data = np.loadtxt("data2Class.txt")
print("data.shape:", data.shape)

np.savetxt("tmp.txt", data) #save data

#split into X data and Y data(labels)
X, y = data[:, :2], data[:, 2]
print("X,shape:", X.shape)
print("y.shape:", y.shape)

# 3D plotting

#get the feature, since linear features phi(x) = x
X_fea = X



#discriminative function
beta = np.zeros(X_fea.shape[1])    # can we first initialize beta = 0? since we have the result, yes
flag = True

while flag:
    print("beta:", beta)
    #class probability
    p = findClassPro(X_fea,beta)

    #neg-log-likelihood gradient
    vlambda = 1e-4
    print("lambs:", vlambda)
    L_gra = dot(X_fea.T,(p-y))+ 2*vlambda*dot(np.eye(beta.shape[0]),beta)
    print("gra shape:", L_gra.shape)
    #hessian
    W = np.diag(dot(np.diag(p),(np.ones(X_fea.shape[0])-p)))
    #print("w:",dot(W,X))
    L_hes = dot(X_fea.T,dot(W,X_fea))+2*vlambda*np.eye(X_fea.shape[1])

    print("hes shape:", L_hes.shape)

    #newton iteration
    temp = beta
    beta = beta - dot(inv(L_hes),L_gra)

    print("gradient:",dot(inv(L_hes),L_gra))
    if (temp==beta).all():
        flag = False

optbeta = beta
print("final beta:",optbeta)


#2d plotting
X_grid = grid2d(-3, 3, num=30)
p_plt = findClassPro(X_grid,optbeta)

# start plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_grid[:,0],X_grid[:,1],p_plt, color='red')
ax.scatter(X[:,0],X[:,1],y)
ax.set_title("predicted probability")
plt.show()

