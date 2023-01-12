import numpy as np
import scipy.io as sio

from scipy.spatial import distance_matrix


def cluster_dist(c1, c2):
    dist_mat = distance_matrix(c1, c2)
    return np.min(dist_mat)

def singlelinkage(X, k):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    clusters = [[x] for x in X]
    while len(clusters) > k:
        print(len(clusters))
        indices = (0, 1)
        min_cluster_dist = cluster_dist(clusters[0], clusters[1])
        for i in range(len(clusters)):
            for j in range(i+1, len(clusters)):
                dist = cluster_dist(clusters[i], clusters[j])
                if dist < min_cluster_dist:
                    min_cluster_dist = dist
                    print("--")
                    indices = (i, j)
        c1 = clusters[indices[0]]
        c2 = clusters[indices[1]]
        c1.extend(c2)
        del clusters[indices[1]]

    labels = np.ones(len(X)) * -1
    for i, x in enumerate(X):
        found = False
        for j in range(len(clusters)):
            for v in clusters[j]:
                if np.array_equal(x, v):
                    print(i / len(X), "%")
                    labels[i] = j
                    found = True
                    break
            if found:
                break
    return labels.reshape(len(X), 1)



def singlelinkage2(X, k):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    clusters = np.array(X)
    clusters.reshape()




def single_linkage_test():
    data = np.load('mnist_all.npz')
    X = data["train0"]
    true_labels = np.zeros(len(X))
    for i in range(1, 10):
        X = np.concatenate((X, data[f"train{i}"]))
        true_labels = np.concatenate((true_labels, i * np.ones(len(data[f"train{i}"]))))
    k = 10
    t = 100
    indices = np.random.randint(0, 60000, 300)
    sample = []
    sample_labels = []
    for i in range(len(indices)):
        sample.append(X[indices[i]])
        sample_labels.append(true_labels[indices[i]])

    sample = np.array(sample)
    sample_labels = np.array(sample_labels)
    clusters = singlelinkage(sample, k)
    clusters_counters = np.zeros((10, 10))
    for i in range(k):
        cluster_size = 0
        labels_in_cluster = np.zeros(10)
        for j in range(len(sample_labels)):
            if clusters[j] == i:
                cluster_size += 1
                labels_in_cluster[int(sample_labels[j])] += 1
        most_common_label = np.argmax(labels_in_cluster)
        print(f"#{i}:   size = {cluster_size}, most common label = {most_common_label}, appeared = {labels_in_cluster[most_common_label] / cluster_size }")



def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    X = np.concatenate((data['train0'], data['train1']))
    m, d = X.shape

    # run single-linkage
    c = singlelinkage(X, k=10)

    assert isinstance(c, np.ndarray), "The output of the function softsvm should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"





if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    single_linkage_test()

    # here you may add any code that uses the above functions to solve question 2
