import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as plt
import scipy

#part (a)
n = 24.0
p = 10


temperatures = np.array([75.0, 77.0, 76.0, 73.0, 69.0, 68.0, 63.0, 59.0,
57.0, 55.0, 54.0, 52.0, 50.0, 50.0, 49.0, 49.0,
49.0, 50.0, 54.0, 56.0, 59.0, 63.0, 67.0, 72.0 ])
ts = np.arange(0,n)+1

#fapprox = scipy.interpolate.interp1d(ts, temperatures)
#hours = np.arange(0.0,n,0.01)
hours = np.copy(ts)
#print(hours)

f = np.copy(temperatures).T #fapprox(hours)#np.copy(temperatures).T
#print(temp.shape, hours.shape)
#print(hours)

X = []
for i in range(p+1):
    X.append(hours**i)
X = np.array(X).T

#print(X.shape)
#print(X[:,0:3])

#Obtain least squares solution
alpha_L2 = np.linalg.pinv(X) @ f
f_L2 = X @ alpha_L2

#Obtain LASSO solution
lasso_mod = linear_model.Lasso(alpha=1.0)# ,fit_intercept=True, normalize='deprecated')
lasso_mod.fit(X,f)
alpha_lasso = lasso_mod.coef_
f_lasso = X @ alpha_lasso

#compute ridge regression solution
ridge_mod = linear_model.Ridge(alpha=1.0)#,normalize=True)
ridge_mod.fit(X,f)
alpha_ridge = ridge_mod.coef_
f_ridge = X @ alpha_ridge

#compute ElasticNet solution
enet_mod = linear_model.ElasticNet(alpha=1.0,random_state=0)#l1_ratio=0.5, fit_intercept=True, normalize='deprecated')
enet_mod.fit(X,f)
alpha_eNet= enet_mod.coef_
f_eNet = X @ alpha_eNet


e0 = np.linalg.norm(f-f_L2,ord=2)/np.linalg.norm(f,ord=2)
e1 = np.linalg.norm(f-f_lasso,ord=2)/np.linalg.norm(f,ord=2)
e2 = np.linalg.norm(f-f_ridge,ord=2)/np.linalg.norm(f,ord=2)
e3 = np.linalg.norm(f-f_eNet,ord=2)/np.linalg.norm(f,ord=2)

print('relative errors... (LSS, LASSO, Ridge, ElasticNet)')
print(e0)
print(e1)
print(e2)
print(e3)


fig = plt.figure()
plt.plot(hours, f, color='b',label='True temps')
plt.plot(hours, f_L2, color='r',label='LSS')
plt.plot(hours, f_lasso, color='g',label='LASSO')
plt.plot(hours, f_ridge, color='k',label='RIDGE')
plt.plot(hours, f_eNet, color='pink',label='Elastic Net')
plt.ylabel('Temperature')
plt.xlabel('Time (hours)')
plt.legend()
plt.show()
