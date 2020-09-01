from sklearn.decomposition import non_negative_factorization
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

def sk_kmeans():
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

    nmf_centroids = torch_nmf(X, 3, n_epoch = 1000, ln_rate = 1e-5)
    print("Dimensions")
    print(nmf_centroids)
    print("#rows:{0}".format(nmf_centroids.shape[0]))
    print("#cols:{0}".format(nmf_centroids.shape[1]))
    plt.subplot(223)
    plt.plot(nmf_centroids[:, 0], nmf_centroids[:, 1], 'o', color='black')
    plt.title("NMF centroids")

    plt.show()


prox_plus = nn.Threshold(0,0) ## to make all output postive

class NMF(nn.Module):
    def __init__(self, n, p, num_components, seed=101):
        super(NMF, self).__init__()
        print("{0} obs with {1} features".format(n,p))
        torch.manual_seed(seed)
        self.U = nn.Parameter(torch.randn(n, num_components, requires_grad=True))
        self.V = nn.Parameter(torch.randn(num_components, p, requires_grad=True))

    def forward(self):
        return self.U, self.V


def torch_nmf(X, k,n_epoch = 1000, ln_rate = 1e-2):
    n = X.shape[0]
    p = X.shape[1]
    beta = 0.2
    nmf = NMF(n, p, k)

    loss_fn = nn.MSELoss(reduction='sum')
    ttloss = []
    optimizer = optim.SGD(nmf.parameters(), lr=ln_rate)
    X_ = torch.from_numpy(X).to(torch.float32)

    for epoch in range(n_epoch):
        U, V = nmf()
        X_hat = torch.matmul(U, V)
        #loss = torch.norm(X_ - X_hat, p = 'fro')
        loss = loss_fn(X_hat, X_) + beta * torch.square(torch.norm(V, dim=0)).sum()
        nmf.zero_grad()
        loss.backward()
        optimizer.step()
        ttloss.append(loss)

        #for param in nmf.parameters():
        #    param.data = prox_plus(param.data - ln_rate * param.grad)

    newU = nmf.U.detach().cpu().numpy()
    newV = nmf.V.detach().cpu().numpy()
    #print(np.dot(newU, newV).transpose())
    return newU, newV, ttloss[-1]

if __name__ == '__main__':
    X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    U,V, loss_value = torch_nmf(X, 2)
    print("\tX")
    print(X)
    print("\tU")
    print(U)
    print("\tV")
    print(V)
    print("\t<U.V> last loss value:{0:.3f}".format(loss_value))
    print(np.dot(U,V))

if __name__ == '__main__1':
    sk_kmeans()

if __name__ == '__main2__':
    X = np.array([[1,1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    W, H, n_iter = non_negative_factorization(X, n_components=2, init='random', random_state=0)
    print("#Iters.",n_iter)
    print(X)
    print("\tW")
    print(W)
    print("\tH")
    print(H)
    print('Checking (sklearn)...')
    print(np.dot(W,H))

    # pytorch implementation
    print('PyTorch implementation')
    n = X.shape[0]
    p = X.shape[1]
    k = 2

    nmf = NMF(n, p, k)
    n_epoch = 1000
    ln_rate = 1e-2
    loss_fn = nn.MSELoss(reduction='sum')
    ttloss = []
    optimizer = optim.SGD(nmf.parameters(), lr=ln_rate)
    X_ = torch.from_numpy(X).to(torch.float32)
    print("Initial matrix(torch version)")
    print(X_)

    for epoch in range(n_epoch):
        X_hat = nmf()
        #loss = torch.norm(X_ - X_hat, p = 'fro')
        loss = loss_fn(X_hat, X_)
        nmf.zero_grad()
        loss.backward()
        optimizer.step()
        ttloss.append(loss)

        #for param in nmf.parameters():
        #    param.data = prox_plus(param.data - ln_rate * param.grad)

    print('Learning curve for task')
    plt.plot(ttloss)
    plt.ylabel('loss over time')
    plt.xlabel('iteration times')
    plt.show()
    print('Final loss')
    print(ttloss[-1])
    print("\tU")
    print(nmf.U)
    print("\tV")
    print(nmf.V)
    # checking
    print('Checking...')
    print(prox_plus(torch.matmul(nmf.U, nmf.V )))
    newU = nmf.U.detach().cpu().numpy()
    newV = nmf.V.detach().cpu().numpy()
    print(np.dot(newU, newV).transpose())











