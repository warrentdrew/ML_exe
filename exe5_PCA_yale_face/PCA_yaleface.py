import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.sparse.linalg as sla
from numpy import linalg as la  #try this method
from PIL import Image
import scipy as sp

def retrievePic(X,index,h,w):
    temp  = X[index,:w]
    for i in range(1,h):
        temp = np.row_stack([temp,X[index,i*w:((i+1)*w)]])
    print("retrieve rigin image:",temp.shape)
    return temp

#img_test = plt.imread("yalefaces/subject01.happy")
#plt.imshow(img_test)
#plt.show()
firstFlag = True;

directory = os.fsencode("yalefaces")
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    filepath = "yalefaces/"+filename
    img = plt.imread(filepath)
    img_vec = img.flatten()
    #print("sdfdfsdf",img.shape)
    if firstFlag:
        X = img_vec
        firstFlag = False
        height = img.shape[0]
        width = img.shape[1]
    else:
        X = np.row_stack([X,img_vec])


print("shape:", X.shape)
sum = 0
for i in range(X.shape[0]):
    sum = sum + X[i,:]

n = X.shape[0]
mu = sum/n

print(mu)
print("mu_shape:",mu.shape)

A = np.mat(np.ones(X.shape[0])).T   #1_n
X_ = X - np.dot(np.mat(A),np.mat(mu))
print("X_",X_.shape)

#sigular value decomposition
num_sigularvalues = min(X_.shape)-1   #find out the number of sigular values need to preserve
                        #e.g. this number is set to 60
                        #s will have 60 values, this means u will be 166*60 and vt will be 60*77760
                        #num_sigularvalue
#print("numadfsd",num_sigularvalues)
u, s, vt = sla.svds(X_, num_sigularvalues)        #be careful not to omit the second parameter as the default value is 6,
                                            # which is absolutely not enough for this problem
                                            #this method changes s into square matrix dimension is the num_singularvalues=165
print("s:",s)
#find Z in PCA
p = 140
V_op = vt.T[:,:p]
print("shape vop:", V_op.shape)
Z = np.dot(X_,V_op)
print("Z shape:",Z.shape)

#reconstruction
X_re = np.dot(A,np.mat(mu)) + np.dot(Z,V_op.T)

#retrieve the original picture with idx
idx = 3
img_ori = retrievePic(X_,idx,height,width)

#retrieve reconstructed image
img_rec = retrievePic(X_re, idx, height, width)


#print("error:",np.linalg.norm())

plt.figure()
plt.subplot(1,2,1)
plt.title("original image")
plt.imshow(img_ori)

plt.subplot(1,2,2)
plt.title("reconstructed image")
plt.imshow(img_rec)

plt.show()