import torch
import numpy as np
from torchnmf import NMF

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def nmf_vs_kmeans():
    plt.figure(figsize=(12, 12))

    n_samples = 1500
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)

    # Incorrect number of clusters
    y_pred = KMeans(n_clusters=2, random_state=random_state).fit_predict(X)
    plt.subplot(221)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("Incorrect Number of Blobs")

    y_pred = KMeans(n_clusters=3, random_state=random_state).fit_predict(X)

    plt.subplot(222)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title("Anisotropicly Distributed Blobs")

    W, H = nmf_gpu(X, 3, max_iter=200) # focus on H (num. representatives x num. features)
    nmf_centroids = H
    print("Dimensions")
    print(nmf_centroids)
    print("#rows:{0}".format(nmf_centroids.shape[0]))
    print("#cols:{0}".format(nmf_centroids.shape[1]))
    plt.subplot(223)
    plt.plot(nmf_centroids[:, 0], nmf_centroids[:, 1], 'o', color='black')
    plt.title("NMF centroids")

    plt.show()

def nmf_gpu(npArr, k, max_iter=100):
    S = torch.FloatTensor(npArr)
    R = k
    net = NMF(S.shape, rank=R).cuda()
    _, V = net.fit_transform(S.cuda(), verbose=True, max_iter=max_iter)
    # net.sort()
    W, H = net.W.detach().cpu().numpy(), net.H.detach().cpu().numpy()
    V = V.detach().cpu().numpy()
    return W, H

if __name__ == '__main__':
    nmf_vs_kmeans()

if __name__ == '__main2__':
    S = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    W,H = nmf_gpu(S, 3)
    print("Input data #rows:{0} #cols:{1}".format(S.shape[0], S.shape[1]))
    """
    S = torch.FloatTensor(S)
    R = 2
    net = NMF(S.shape, rank=R).cuda()
    _, V = net.fit_transform(S.cuda(), verbose=True, max_iter=100)
    #net.sort()
    W, H = net.W.detach().cpu().numpy(), net.H.detach().cpu().numpy()
    V = V.detach().cpu().numpy()
    """
    print("W #rows:{0} #cols:{1}".format(W.shape[0], W.shape[1]))
    print("H #rows:{0} #cols:{1}".format(H.shape[0], H.shape[1]))

    print("\tS")
    print(S)
    print("\tW")
    print(W)
    print("\tH")
    print(H)
    #print("\tV") # same as the product between W and H
    #print(V)
    print('\nChecking decomposition...\n')
    print(np.dot(W, H))