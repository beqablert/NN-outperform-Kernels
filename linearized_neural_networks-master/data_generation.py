import numpy as np
import os
import matplotlib.pyplot as plt

def generate_nonuniform_data(n, d, eta, kappa, index):
    # The dimension of the relevant Region
    d1 = np.int64(d ** eta)
    nt = 10000    
    exponent = (eta + kappa) / 2.0
    r = d ** exponent
    print(r'd_1 is %d'%(d1))
    print(r'Kappa is %f'%(kappa))
    print(r'The radius $\sqrt{d}$ is %f'%(np.sqrt(d)))
    print('The radius r is %f'%(r))
    
    # Making the features
    np.random.seed(145)
    # Train Data
    X = np.random.normal(size=(n, d))
    X = X.astype(np.float32)
    for i in range(n):
        X[i, :d1] = X[i, :d1] / np.linalg.norm(X[i, :d1]) * r
        X[i, d1:] = X[i, d1:] / np.linalg.norm(X[i, d1:]) * np.sqrt(d)

    # Test Data
    np.random.seed(2)
    XT = np.random.normal(size=(nt, d))
    XT = XT.astype(np.float32)
    for i in range(nt):
        XT[i, :d1] = XT[i, :d1] / np.linalg.norm(XT[i, :d1]) * r
        XT[i, d1:] = XT[i, d1:] / np.linalg.norm(XT[i, d1:]) * np.sqrt(d)
        
    directory = './datasets/synthetic/'
    np.save(directory + 'X_train_anisotropic_%d_%d_%d.npy'%(d, d1, index), X)
    np.save(directory + 'X_test_anisotropic_%d_%d_%d.npy'%(d, d1, index), XT)
    X0 = X[:, :d1]
    X1 = XT[:, :d1]
    del X, XT
    
    # Make the labels
    np.random.seed(14)
    f = []
    # The function has no linear component
    beta2 = np.random.exponential(scale=1.0, size=(d1 - 1, 1))
    beta3 = np.random.exponential(scale=1.0, size=(d1 - 2, 1))
    beta4 = np.random.exponential(scale=1.0, size=(d1 - 3, 1))
        
    Z = np.multiply(X0[:, :-1], X0[:, 1:])
    temp = np.dot(Z, beta2)
    f.append(temp)

    Z = np.multiply(X0[:, :-2], X0[:, 1:-1])
    Z = np.multiply(Z, X0[:, 2:])
    temp = np.dot(Z, beta3)
    f.append(temp)

    Z = np.multiply(X0[:, :-3], X0[:, 1:-2])
    Z = np.multiply(Z, X0[:, 2:-1])
    Z = np.multiply(Z, X0[:, 3:])
    temp = np.dot(Z, beta4)
    f.append(temp)
    
    normalization = [np.sqrt(np.mean(t ** 2)) for t in f]
    for i in range(len(f)):
        f[i] = f[i] / normalization[i]
        
    totalf = f[0] + f[1] + f[2]
    totalf = totalf.astype(np.float32)
    
    g = []
    
    Z = np.multiply(X1[:, :-1], X1[:, 1:])
    temp = np.dot(Z, beta2)
    g.append(temp)

    Z = np.multiply(X1[:, :-2], X1[:, 1:-1])
    Z = np.multiply(Z, X1[:, 2:])
    temp = np.dot(Z, beta3)
    g.append(temp)

    Z = np.multiply(X1[:, :-3], X1[:, 1:-2])
    Z = np.multiply(Z, X1[:, 2:-1])
    Z = np.multiply(Z, X1[:, 3:])
    temp = np.dot(Z, beta4)
    g.append(temp)
    for i in range(len(g)):
        g[i] = g[i] / normalization[i]
    totalg = g[0] + g[1] + g[2]
    totalg = totalg.astype(np.float32)
    
    np.save(directory + 'y_train_anisotropic_%d_%d_%d.npy'%(d, d1, index), totalf)
    np.save(directory + 'y_test_anisotropic_%d_%d_%d.npy'%(d, d1, index), totalg)

d = 128
eta = 2.0 / 5.0
n = 1024 * 1024
kappa_mat = np.linspace(0, 1, num=10, endpoint=False)

for index in range(10):
    generate_nonuniform_data(n, d, eta, kappa_mat[index], index)