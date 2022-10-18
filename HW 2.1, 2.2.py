# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 16:05:26 2022

@author: Admin
"""
import numpy as np
import matplotlib.pyplot as plt

Dog = plt.imread('dog.jpg')
Dog = np.mean(Dog, -1) #convert RGB to grayscale
plt.figure(figsize=(15,10))
plt.imshow(Dog, cmap='gray')
plt.axis('off');

Dog_fft = np.fft.fft2(Dog)
Dog_fftsort = np.sort(np.abs(Dog_fft.reshape(-1)))

error = []

for keep in range(1,101):
    thresh = Dog_fftsort[int(np.floor((1-keep/100)*len(Dog_fftsort)))]
    ind = np.abs(Dog_fft)>thresh          # Find small indices
    Dog_fftlow = Dog_fft * ind                 # Threshold small indices
    Dog_low = np.fft.ifft2(Dog_fftlow).real  # Compressed image
    error.append(((Dog - Dog_low)**2).mean())
    
plt.figure(figsize=(15,5))
plt.plot(error, color='black')
plt.xlabel('Compressed ratio, %')
plt.ylabel('MSE')
plt.title('Error between Compressed and Actual Images');

dx = 0.02
L = 2
x = L * np.arange(-1+dx,1+dx,dx)
n = 100
nquart = int(np.floor(n/4))

f = np.zeros_like(x)
for l in range(nquart,3*nquart - 1):
    f[l] = 1-np.abs(x[l])

A0 = 1/4
An = np.zeros(n)
Bn = 0
fFS = A0
    
fig, ax = plt.subplots(figsize=(15,8))
ax.plot(x,f,'-',color='r',linewidth=2)
name = "Accent"
cmap = plt.get_cmap('tab10')
colors = cmap.colors
ax.set_prop_cycle(color=colors)

for k in range(n):
    An[k] = (4-4*np.cos((np.pi*(k+1))/2))/((np.pi**2)*((k+1)**2))
    fFS = fFS + An[k]*np.cos(((k+1)*np.pi*x)/L) + Bn*np.sin(((k+1)*np.pi*x)/L)
    ax.plot(x,fFS)

fig, ax = plt.subplots(1, 2, figsize=(18,5))

ax[0].plot(An, color='r')
ax[0].set_title('An coefficients')
ax[1].plot(np.zeros(100), color='r')
ax[1].set_title('Bn coefficients');