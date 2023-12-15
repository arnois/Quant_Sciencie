# -*- coding: utf-8 -*-
"""

@author: arnulfqc
"""
###############################################################################
# MODULES
import numpy as np
import datetime as dt
from matplotlib import pyplot as plt

from sklearn import datasets
from sklearn.metrics.pairwise import euclidean_distances as d_ED
from sklearn.cluster import KMeans

# UDF
## Gaussian kernel
def gaussian_kernel(x, sigma=1.0): return np.exp(-x**2 / (2 * sigma**2))
## Support matrix for gradient(H)
def eD2(D,sigma):
    eD = gaussian_kernel(D,sigma)
    D2 = D**2/(sigma**3)
    E = np.multiply(D2,eD)
    return E
## Gradient of H = trace(Q.T@S@Q)
def gradientG(Q,D,sigma):
    E = eD2(D,sigma)
    G = 0
    for i in range(Q.shape[1]):
        q_i = Q[:,i].reshape(-1,1)
        G += (q_i.T @ E @ q_i).item(0)
    
    return G
## Solve max H=trace(Q.T@S@Q) for Q
def solve_for_Q(sigma,S,K):
    S = gaussian_kernel(D,sigma);np.fill_diagonal(S,0)
    U = np.diag(np.sum(S,axis=1))
    S_norm = np.linalg.inv(U)@S
    w, v = np.linalg.eig(S_norm)
    
    return v[:,:K]

# SV
colors = np.array(["#377eb8", "#ff7f00", "#4daf4a", "#f781bf", 
                   "#a65628", "#984ea3", "#999999", "#e41a1c", "#dede00"])

###############################################################################
# Following Wang & Zhang
# Input
#blobs = datasets.make_blobs(n_samples=1000, n_features=4)
noisy_moons = datasets.make_moons(1000, noise=0.05)
X,y = noisy_moons
plt.scatter(x = X[:,0], y = X[:,1], color=colors[y])

X = X.T
err = 1e-4
sigma0 = 1
alpha=0.05

# Step 1
# Similarity Measure - Euclidean distance (ED)
D = d_ED(X.T, X.T);np.fill_diagonal(D,0)

# Step 2
# Affinity matrix
S = gaussian_kernel(D,1);np.fill_diagonal(S,0)

# Step 3
# Spectral decomposition
U = np.diag(np.sum(S,axis=1))
S_norm = np.linalg.inv(U)@S
w, v = np.linalg.eig(S_norm) #;plt.plot(w[:20])
n_clusts = np.sum(abs(w-1)<0.1) # eigenvalue count around 1
print(f'Clusters found: {n_clusts}')
Q_0 = v[:,:n_clusts] # init guess for Q in max trace(Q.T@S@Q)
# enforced by user: n_clusts = 2

# Step 4: determine sigma
t1 = dt.datetime.now()
T = 50
sigma1 = sigma0
for i in range(T):
    # Solve for H to get Q
    Qstar = solve_for_Q(sigma1,S,n_clusts)
    # Compute gradient (H)
    grad = gradientG(Qstar,D,sigma1)
    # Update variance
    if abs(grad)<err:
        break
    else:
        sigma1 += alpha*grad
t2 = dt.datetime.now()
print(f'Gauss kernel variance tune time: {(t2-t1).seconds/60:.1f} min')        

# Step 5: Cluster Qstar
kmeans = KMeans(n_clusters = n_clusts, random_state=13).fit(Qstar)

# Assign features to cluster # clust_set
clust_set = {}
for k in range(n_clusts):
    clust_set[k] = np.where(kmeans.labels_ == k)[0].tolist()
y_hat = np.zeros(X.shape[1])
y_hat[clust_set[1]] = 1
#y_hat[clust_set2[2]] = 2
y_hat = y_hat.astype(int)

plt.scatter(x = X[0,:], y = X[1,:], color=colors[y_hat])
