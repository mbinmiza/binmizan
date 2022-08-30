# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 12:35:29 2022

@author: Admin
"""

from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = [16, 8]


A = imread(os.path.join('dog.jpg'))
X = np.mean(A, -1); # Convert RGB to grayscale

img = plt.imshow(X)
img.set_cmap('gray')
plt.axis('off')
plt.show()


U, S, VT = np.linalg.svd(X)
U1=U.T #Transpose of U

IM=U1@U   #U*U
print(np.around(IM[:15,:15]))
IM2=U@U1
print(np.around(IM[:15,:15]))
idm=np.identity(len(U))
print (idm)
mse = ((IM2 - idm)**2)
print(mse)
plt.plot(mse)
plt.show()