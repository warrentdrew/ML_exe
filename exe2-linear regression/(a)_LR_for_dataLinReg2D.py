"""
this is a mini demo for using numpy and arrays and plot data
"""
# install numpy package
from functools import reduce

import numpy as np


from numpy import dot
from numpy.linalg import inv

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D  # 3D ploting


# Helper function
def prepend_one(X):
    return np.column_stack([np.ones(X.shape[0]), X])

def mdot(*args):
    return reduce(np.dot, args)

def grid2d(start, end, num=50):
    dom = np.linspace(start,end,num)
    X0, X1 = np.meshgrid(dom,dom)
    return np.column_stack([X0.flatten(), X1.flatten()])

"""
data = np.loadtxt("dataLinReg2D.txt")
print("data.shape:", data.shape)
np.savetxt("tmp.txt", data)
X, y = data[:, :2], data[:, 2]

a = np.array([[1, 3], [2, 4]])
print(a)

print("X.shape:", X.shape)
print("y.shape:", y.shape)

"""

"""

# 3D plotting
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y, color="red")
ax.set_title("raw data")

plt.draw()
# plt.show()

#prep for linear reg
X = prepend_one(X)
print("X.shape:", X.shape)


#Fit model/compute optimal parameters beta
#beta_ = mdot(inv(dot(X.T, X)), X.T, y)
#print("Optimal beta:", beta_)


"""
"""
#for exe b)
# compute quadratic feature
#use the upper triangle of the X.T * X
data = np.loadtxt("dataQuadReg2D.txt")
print("dataQuad.shape:", data.shape)
np.savetxt("tmp.txt", data)
X, y = data[:, :2], data[:, 2]

X = prepend_one(X)
print("X.shape:", X.shape)
"""

def quadFeature(X):
    quadmatrix = []
    temp = []
    for j in range(X[0].shape[0]):
        quadmatrix = np.append(quadmatrix,dot(X[0,j],X[0,j:]))

    for i in range(1,(X.shape[0])):
        for j in range(X[i].shape[0]):
            temp = np.append(temp,dot(X[i,j],X[i,j:]))
        quadmatrix = np.row_stack([quadmatrix,temp])
        temp = []
    #print("quadmatrix:",quadmatrix)
    return quadmatrix




data = np.loadtxt("dataQuadReg2D_noisy.txt")
print("dataQuad_noisy.shape:", data.shape)
np.savetxt("tmp.txt", data)
X, y = data[:, :2], data[:, 2]

X = prepend_one(X)
print("X.shape:", X.shape)

X = quadFeature(X)

#for exe3 find the mse and deviation for different regression parameters "lambda"
def computeError(X,y,reg):
    arr = np.eye(X.T.shape[0])
    arr[0,0] = 0

    #cross validation
    numeachset = 10
    setnum = int(X.shape[0]/numeachset)
    #print("number of sets:", setnum)
    #print("number each set:", numeachset)
    sumerr = 0
    squaresumerr = 0

    for i in range(setnum):
        X_temp = X[10*i:10*(1+i)-1,:]
        y_temp = y[10*i:10*(1+i)-1]
        new_beta = mdot(inv(dot(X_temp.T, X_temp)+dot(reg,arr)), X_temp.T, y_temp)
        #print("New beta:", new_beta)
        newSquareError = (np.linalg.norm(y_temp-dot(X_temp,new_beta))**2)/numeachset
        #print("square error for",i, "is: ",newSquareError)
        sumerr = sumerr+newSquareError
        squaresumerr = squaresumerr + newSquareError**2
        #print("sumerr:",sumerr)

    meanSquareError = sumerr/setnum
    #print("mean square error:", meanSquareError)

    deviation = np.sqrt((squaresumerr - setnum * meanSquareError ** 2) / (setnum - 1))
    #print("deviation:",deviation)
    print("lambda=",reg," ",meanSquareError," ",deviation)

for regparam in range(0,500,10):
    #print("reg:",regparam)
    computeError(X,y,regparam)


arr = np.eye(X.T.shape[0])
arr[0,0] = 0
#print("Optimal beta:", beta_)
beta_ = mdot(inv(dot(X.T, X)), X.T, y)

# report squared error （this should consider the X of training data）
squareError = np.linalg.norm(y-dot(X,beta_))**2
#print("square error: ",squareError)

#prep for prediction
X_grid = prepend_one(grid2d(-3, 3, num=30))
#print("X_grid.shape:", X_grid.shape)

X_grid = quadFeature(X_grid)
#print("Quad X_grid.shape:", X_grid.shape)

#predict with trained model
y_grid = mdot(X_grid, beta_)
#print("Y_grid.shape", y_grid.shape)



#vis the result
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_grid[:,1], X_grid[:,2], y_grid)
ax.scatter(X[:, 1],X[:,2], y, color="red")
ax.set_title("predicted data")
plt.show()

