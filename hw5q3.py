import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def buggy_PCA(X, d):
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    z = np.dot(X, vt[:d].T)
    reconstruction = np.dot(z, vt[:d])
    return z, reconstruction, u, s, vt[:d]


def demeaned_PCA(X, d):
    means = np.mean(X, axis=0)
    X = X - means
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    z = np.dot(X, vt[:d].T)
    reconstruction = np.dot(z, vt[:d]) + means
    return z, reconstruction, u, s, vt[:d]


def normalized_PCA(X, d):
    means = np.mean(X, axis=0)
    X = X - means
    std = np.std(X, axis=0)
    X = X / std
    u, s, vt = np.linalg.svd(X, full_matrices=False)
    z = np.dot(X, vt[:d].T)
    reconstruction = np.dot(z, vt[:d]) * std + means
    return z, reconstruction, u, s, vt[:d]


def reconstruction_error(X, reconstruction):
    return np.sum((X - reconstruction)**2)


data_2d = pd.read_csv("data/data2D.csv", header=None).values
data_1000d = pd.read_csv("data/data1000D.csv", header=None).values

reconstruction_errors = [["Approach", "2D", "1000D"]]
# Buggy PCA
z, reconstruction, u, s, vt = buggy_PCA(data_2d, 1)
z, reconstruction_1000, u, s, vt = buggy_PCA(data_1000d, 30)

reconstruction_errors.append(["buggy_PCA", reconstruction_error(
    data_2d, reconstruction), reconstruction_error(data_1000d, reconstruction_1000)])


plt.scatter(data_2d[:, 0], data_2d[:, 1],
            marker="o", facecolor="none", edgecolor="b", label="original")
plt.scatter(reconstruction[:, 0], reconstruction[:, 1],
            marker="+", c="r", label="reconstructed")
plt.title("Buggy PCA")
plt.legend()
plt.savefig("buggy_pca.png")
plt.clf()

# Demeaned PCA
z, reconstruction, u, s, vt = demeaned_PCA(data_2d, 1)
z, reconstruction_1000, u, s, vt = demeaned_PCA(data_1000d, 30)

reconstruction_errors.append(["demeaned_PCA", reconstruction_error(
    data_2d, reconstruction), reconstruction_error(data_1000d, reconstruction_1000)])

plt.scatter(data_2d[:, 0], data_2d[:, 1],
            marker="o", facecolor="none", edgecolor="b", label="original")
plt.scatter(reconstruction[:, 0], reconstruction[:, 1],
            marker="+", c="r", label="reconstructed")
plt.title("Demeaned PCA")
plt.legend()
plt.savefig("demeaned_pca.png")
plt.clf()

# Normalized PCA
z, reconstruction, u, s, vt = normalized_PCA(data_2d, 1)
z, reconstruction_1000, u, s, vt = normalized_PCA(data_1000d, 30)

reconstruction_errors.append(["normalized_PCA", reconstruction_error(
    data_2d, reconstruction), reconstruction_error(data_1000d, reconstruction_1000)])

plt.scatter(data_2d[:, 0], data_2d[:, 1],
            marker="o", facecolor="none", edgecolor="b", label="original")
plt.scatter(reconstruction[:, 0], reconstruction[:, 1],
            marker="+", c="r", label="reconstructed")
plt.title("Normalized PCA")
plt.legend()
plt.savefig("normalized_pca.png")
# plt.show()
plt.clf()

print(reconstruction_errors)

pca = PCA()
pca.fit(data_1000d)
variance = pca.explained_variance_ratio_
eigenvalues = pca.explained_variance_

plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.xlabel('Component Number')
plt.ylabel('Eigenvalue')
plt.title('Scree Chart')
plt.grid(True)
plt.xlim(0, 50)
plt.savefig("scree.png")
