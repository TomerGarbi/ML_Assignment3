import numpy as np



def random_centroids(k, X):
    Temp = np.array(X)
    np.random.shuffle(Temp)
    return Temp[:k]



def compare_arrays(x1, x2):
    for i in range(len(x1)):
        if x1[i] != x2[i]:
            return False
    return True


def kmeans(X, k, t):
    """
    :param X: numpy array of size (m, d) containing the test samples
    :param k: the number of clusters
    :param t: the number of iterations to run
    :return: a column vector of length m, where C(i) ∈ {1, . . . , k} is the identity of the cluster in which x_i has been assigned.
    """
    centroids = random_centroids(k, X)
    map = [[] for _ in centroids]
    iter = 0
    while iter < t:
        new_map = [[] for _ in centroids]
        for i, x in enumerate(X):
            lengths = []
            for c in range(k):
                lengths.append(np.linalg.norm(centroids[c] - x))
            new_map[np.argmin(lengths)].append(x)
        map = new_map
        iter += 1

        for i in range(len(map)):
            centroids[i] = sum(map[i]) / len(map[i])

    labels = np.ones(len(X)) * -1
    for i, x in enumerate(X):
        found = False
        for j in range(len(map)):
            for v in map[j]:
                if np.array_equal(x, v):
                    print(i/len(X), "%")
                    labels[i] = j
                    map[j].remove(v)
                    found = True
                    break
            if found:
                break
    return labels.reshape(len(X), 1)


def simple_test():
    # load sample data (this is just an example code, don't forget the other part)
    data = np.load('mnist_all.npz')
    X = np.concatenate((data['train0'], data['train1']))
    m, d = X.shape

    # run K-means
    print(len(X))
    c = kmeans(X, k=10, t=10)
    print(c.shape)
    assert isinstance(c, np.ndarray), "The output of the function kmeans should be a numpy array"
    assert c.shape[0] == m and c.shape[1] == 1, f"The shape of the output should be ({m}, 1)"


def kmeans_test():
    data = np.load('mnist_all.npz')
    X = data["train0"]
    true_labels = np.zeros(len(X))
    for i in range(1, 10):
        X = np.concatenate((X, data[f"train{i}"]))
        true_labels = np.concatenate((true_labels, i * np.ones(len(data[f"train{i}"]))))
    k = 6
    t = 100
    indices = np.random.randint(0, 60000, 1000)
    sample = []
    sample_labels = []
    for i in range(len(indices)):
        sample.append(X[indices[i]])
        sample_labels.append(true_labels[indices[i]])

    sample = np.array(sample)
    sample_labels = np.array(sample_labels)
    clusters = kmeans(sample, k, t)
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















if __name__ == '__main__':
    # before submitting, make sure that the function simple_test runs without errors
    kmeans_test()

    # here you may add any code that uses the above functions to solve question 2
