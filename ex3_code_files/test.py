import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sc


data = sio.loadmat("regdata.mat")

X = data["X"].T
Y = data["Y"]

X_test = data["Xtest"].T
Y_test = data["Ytest"]


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
    sample_sizes = [i for i in range(10, 101, 1)]
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
    plt.plot(sample_sizes, np.round(avg_lambdas), label="minimal error lambda")
    plt.legend()
    plt.xlabel("sample size")
    plt.ylabel("lambda value")
    plt.title("lambda for minimum error")
    plt.show()

#
# def makeVector(x1, x2):
#     a = np.array((x1, x2, 0, 0))
#     a[2] = a[1] ** 2 + a[2] ** 3
#     a[3] = (a[3] - a[1]) ** 2
#     return a.reshape((4, 1))
#
#
# c = makeVector(1, 2)
# d = makeVector(2, 3)
#
# e = np.hstack((c, d))
#
# # A = c @ c.T
# A = e @ e.T
# #
# # print("A:", A)
# #
# # print("eigvals:", np.linalg.eigvals(A))



x1 = 6
x2 = -2
x3 = x1 ** 2 + x2 ** 3
x4 = (x3 - x1) ** 2

y1 = 2
y2 = 6
y3 = y1 ** 2 + y2 ** 3
y4 = (y3 - y1) ** 2


x = np.array([[x1, x2, x3, x4],
             [y1, y2, y3, y4]])




t = 4
A = np.zeros((t, 4))
for xt in A:
    xt[0] = np.random.randint(-3, 3)
    xt[1] = np.random.randint(-3, 3)
    xt[2] = xt[0] ** 2 + xt[1] ** 3
    xt[3] = (xt[2] - xt[1]) ** 2
X = A.T @ A
print(A)
print(X.shape)
print (X)
print(np.linalg.eigvals(X))
