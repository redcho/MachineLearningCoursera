import numpy as np
import pylab as plt
import scipy.optimize as op


def sigmoid(X):
    return 1 / (1 + (np.e ** (-1*X)))


def costFunction(theta, x, y):
    lmb = 1
    J = 0
    m = y.size
    h = sigmoid(x.dot(theta))
    J = ((1./m) * (-y.dot(np.log(h)) - (1 - y).dot(np.log(1 - h)))) + ((lmb/2.*m) * np.power(theta[1:], 2).sum())
    return J


def gradient(theta, x, y):
    lmb = 1
    grad = np.zeros(theta.shape).flatten()
    m = y.size
    h = sigmoid(x.dot(theta))

    for jth in range(x.shape[1]):
        xjth = x[:, jth]
        gradjth = (1./m) * ((h - y) * xjth).sum()
        grad[jth] = gradjth + ((lmb/m) * theta[jth]) if jth != 0 else gradjth

    return grad


def map_feature(x1, x2):
    degree = 6
    out = np.ones(x1.size)

    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.c_[out, np.power(x1, (i-j))*np.power(x2, j)]

    return out


def predict(theta, x):
    p_1 = sigmoid(np.dot(x, theta))
    return p_1 > 0.5

''' Load the data '''
data = plt.loadtxt('ex2data2.txt', delimiter=",")
x = data[:, 0:2]
y = data[:, 2]

''' Plot the data '''
pos = np.where(y == 1)
neg = np.where(y == 0)

plt.scatter(x[pos, 0], x[pos, 1], marker="o", c="b")
plt.scatter(x[neg, 0], x[neg, 1], marker="x", c="r")

plt.xlabel("Microchip Test 1")
plt.ylabel("Microchip Test 2")

plt.legend(['y == 1', 'y == 0'])


featuredX = map_feature(x[:, 0], x[:, 1])
theta = np.zeros(featuredX.shape[1])

''' Optimize the theta using Newton (TNC) algorithm '''
Result = op.minimize(fun=costFunction, x0=theta, args=(featuredX, y), method='TNC', jac=gradient)
opt_theta = Result.x

''' Predict the percentage accuracy of our predictor by feeding it the original dataset '''
g = predict(opt_theta, featuredX)
print(float(y[y == g].size)/float(len(y))*100)