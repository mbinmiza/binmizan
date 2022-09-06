# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 11:52:43 2022

@author: Admin
"""
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os
plt.rcParams['figure.figsize'] = [16, 8]





A = np.random.normal(size=(100,100))
U, S, VT = np.linalg.svd(A)

plt.plot(S)
plt.show()
plt.boxplot(S)
plt.boxplot([S for i in range(100)])

def mdn(S):
    k = []
    for i in range( S.shape[0]):
        k.append(np.median(S[:i+1]))
    return k
for i in [50, 200, 500, 1000]:
    svd_r = np.linalg.svd(np.random.normal(size=(i,i)), full_matrices=True)[1]
    
    plt.ylabel('Singular Values')
    plt.xlabel('r')
    plt.plot(np.cumsum(svd_r)/np.arange(1,len(svd_r)+1))
    plt.plot(np.arange(1,len(svd_r)+1), mdn(svd_r))
   