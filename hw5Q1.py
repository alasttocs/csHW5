import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


def k_means(X, k, restarts=1000):
    best_centroids = None
    best_labels = None
    best_loss = np.inf
    for _ in range(restarts):
        # Step 1: initialize centroids
        centroids = X[np.random.choice(X.shape[0], k, replace=False), :]
        # Step 2: assign labels
        distances = np.zeros((X.shape[0], centroids.shape[0]))
        for i in range(X.shape[0]):
            for j in range(centroids.shape[0]):
                distances[i, j] = np.linalg.norm(X[i] - centroids[j])
        labels = np.argmin(distances, axis=1)
        # Step 3: update centroids
        new_centroids = np.zeros((k, X.shape[1]))
        for i in range(k):
            cluster_points = X[labels == i]
            new_centroids[i] = cluster_points.mean(axis=0)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        loss = np.sum((X - centroids[labels]) ** 2)
        # Step 4: update best centroids
        if loss < best_loss:
            best_loss = loss
            best_centroids = centroids
            best_labels = labels
    return best_centroids, best_labels, best_loss


def gmm(X, k, iterations=100):
    weights = np.full(k, 1 / k)
    means = X[np.random.choice(X.shape[0], k, replace=False), :]
    covariances = np.array([np.cov(X.T) for _ in range(k)])
    best_log_likelihood = -np.inf

    for _ in range(iterations):
        # E-step
        posterior = np.zeros((X.shape[0], k))
        for i in range(k):
            distribution = multivariate_normal(means[i], covariances[i])
            posterior[:, i] = weights[i] * distribution.pdf(X)
        posterior /= posterior.sum(axis=1).reshape(-1, 1)
        # M-step
        total_weight = posterior.sum(axis=0)
        weights = total_weight / X.shape[0]
        old_means = means.copy()
        means = posterior.T.dot(X) / total_weight.reshape(-1, 1)
        for i in range(k):
            distance = X - means[i]
            covariances[i] = (posterior[:, i].reshape(-1, 1) * distance).T.dot(
                distance) / total_weight[i]
        if np.linalg.norm(old_means - means) < .0001:
            break
        # Update best log likelihood
        likelihood = np.zeros((X.shape[0], k))
        for i in range(k):
            distribution = multivariate_normal(means[i], covariances[i])
            likelihood[:, i] = distribution.pdf(X)
        log_likelihood = -np.sum(np.log(np.sum(likelihood, axis=1)))

        if log_likelihood > best_log_likelihood:
            best_log_likelihood = log_likelihood
            best_weights = weights.copy()
            best_means = means.copy()
            best_covariances = covariances.copy()

    return best_weights, best_means, best_covariances, best_log_likelihood


def predict_gmm(X, weights, means, covariances):
    likelihood = np.zeros((X.shape[0], weights.shape[0]))
    for i in range(weights.shape[0]):
        distribution = multivariate_normal(means[i], covariances[i])
        likelihood[:, i] = distribution.pdf(X)
    likelihood /= likelihood.sum(axis=1).reshape(-1, 1)
    return np.argmax(likelihood, axis=1)


# generate data
np.random.seed(0)
sigma = [0.5, 1, 2, 4, 8]

a_mean = np.array([-1, -1])
a_cov = np.array([[2, 0.5], [0.5, 1]])

b_mean = np.array([1, -1])
b_cov = np.array([[1, -0.5], [-0.5, 2]])

c_mean = np.array([0, 1])
c_cov = np.array([[1, 0], [0, 2]])

means = np.array([a_mean, b_mean, c_mean])
synthetic_data = []
for val in sigma:
    p_a = np.random.multivariate_normal(a_mean, val * a_cov, 100)
    p_b = np.random.multivariate_normal(b_mean, val * b_cov, 100)
    p_c = np.random.multivariate_normal(c_mean, val * c_cov, 100)
    synthetic_data.append(np.concatenate((p_a, p_b, p_c)))
synthetic_data = np.array(synthetic_data)

# k-means
k_val = 3
accuracy_list = []
objective_list = []
true_labels_list = []
for i in range(5):
    centroids, labels, loss = k_means(synthetic_data[i], k_val)
    # get the true labels
    closest_points = {}
    for j, centroid in enumerate(means):
        min_dist = np.inf
        closest_points[j] = None
        for k, val in enumerate(centroids):
            dist = np.linalg.norm(val - centroid)
            if dist < min_dist:
                min_dist = dist
                closest_points[j] = k
    true_labels = np.repeat(
        [closest_points[0], closest_points[1], closest_points[2]], 100)
    true_labels_list.append(true_labels)
    accuracy = np.sum(true_labels == labels) / len(true_labels)
    accuracy_list.append(accuracy)
    objective_list.append(loss)
    # print(F"i = {i}")
    # print(f"Sigma = {sigma[i]}")
    # print(f"Loss: {loss}")
    # print(f"Centroids: {centroids}")
    # print(f"closest_points: {closest_points}")
    # print(f"True labels: {true_labels}")
    # print(f"Labels: {labels}")
    # print(f"Accuracy: {accuracy}")
plt.scatter(sigma, accuracy_list)
plt.xlabel("Sigma")
plt.ylabel("Accuracy")
plt.title("K-means Accuracy vs Sigma")
plt.savefig("kmeans1.png")
plt.clf()
plt.scatter(sigma, objective_list)
plt.xlabel("Sigma")
plt.ylabel("Objective")
plt.title("K-means Objective vs Sigma")
plt.savefig("kmeans2.png")
plt.clf()

gmm
k_val = 3
accuracy_list = []
objective_list = []
for i in range(5):
    w, m, c, ll = gmm(synthetic_data[i], k_val)
    labels = predict_gmm(synthetic_data[i], w, m, c)
    closest_points = {}
    for j, point in enumerate(means):
        min_dist = np.inf
        closest_points[j] = None
        for k, val in enumerate(m):
            dist = np.linalg.norm(val - point)
            if dist < min_dist:
                min_dist = dist
                closest_points[j] = k
    true_labels = np.repeat(
        [closest_points[0], closest_points[1], closest_points[2]], 100)
    true_labels_list.append(true_labels)
    accuracy = np.sum(true_labels == labels) / len(true_labels)
    accuracy_list.append(accuracy)
    objective_list.append(ll)
#     print(f"Sigma = {sigma[i]}")
#     print(f"means: {m}")
#     print(f"True labels: {true_labels_list[i]}")
#     print(f"Labels: {labels}")
#     print(f"Accuracy: {accuracy}")
#     print(f"Objective: {ll}")
# print(f"Objective: {objective_list}")
# print(f"Accuracy: {accuracy_list}")
plt.scatter(sigma, accuracy_list)
plt.xlabel("Sigma")
plt.ylabel("Accuracy")
plt.title("GMM Accuracy vs Sigma")
plt.savefig("gmm1.png")
plt.clf()
plt.scatter(sigma, objective_list)
plt.xlabel("Sigma")
plt.ylabel("Objective")
plt.title("GMM Objective vs Sigma")
plt.savefig("gmm2.png")
