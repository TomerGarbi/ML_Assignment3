import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt



data = sio.loadmat("regdata.mat")
print(list(data.keys()))
X = data["X"].T
Y = data["Y"]
print(Y)
X_test = data["Xtest"].T
Y_test = data["Ytest"]

print(X.shape)
print(Y.shape)
print(X_test.shape)
print(Y_test.shape)

def random_sample(X, Y, n):
    indices = np.array([i for i in range(len(X))])
    np.random.shuffle(indices)
    sample = np.array([X[indices[i]] for i in range(n)])
    labels = np.array([Y[indices[i]] for i in range(n)])
    return sample, labels


random_sample(X, Y, 1)




def ridgeRegression(X, y, lambdaRange):
    wList = []
    # Get normal form of `X`
    A = X.T @ X
    # Get Identity matrix
    I = np.eye(A.shape[0])
    # Get right hand side
    c = X.T @ y
    for lambVal in range(lambdaRange, lambdaRange+1):
        # Set up equations Bw = c
        lamb_I = lambVal * I
        B = A + lamb_I
        # Solve for w
        w = np.linalg.solve(B,c)
        wList.append(w)
    return np.array(wList[0])


def ridge_regression():
    lambdas = [i for i in range(31)]
    sample_sizes = [i for i in range(10, 101, 10)]
    avg_lambdas = np.zeros(len(sample_sizes))
    for rep in range(10):
        minimum_lambdas_for_size = []
        for size in sample_sizes:
            min_error = -1
            min_l = -1
            sample, labels = random_sample(X, Y, size)
            for l in lambdas:
                w = np.linalg.inv(sample.T @ sample + l * np.eye(sample.shape[1])) @ sample.T @ labels
                error = 0
                for i, x in enumerate(X_test):
                    error += (w.T.dot(x) - Y_test[i]) ** 2
                if error < min_error or min_error == -1:
                    min_error = error
                    min_l = l
            minimum_lambdas_for_size.append(min_l)
        avg_lambdas += np.array(minimum_lambdas_for_size) / 10
    plt.plot(sample_sizes, np.round(avg_lambdas))
    plt.show()


ridge_regression()
