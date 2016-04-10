from numpy import loadtxt, where, e, log, zeros, dot, ones, c_
from pylab import scatter, show, legend, xlabel, ylabel
import scipy.optimize as op


def sigmoid(X):
    return 1 / (1 + (e ** (-1*X)))


def gradient(theta, x, y):
    grad = zeros(theta.shape).flatten()
    m = y.size
    h = sigmoid(x.dot(theta))

    for jth in range(x.shape[1]):
        xjth = x[:, jth]
        gradjth = (1./m) * ((h - y) * xjth).sum()
        grad[jth] = gradjth

    return grad


def costFunction(theta, x, y):
    J = 0
    m = y.size
    h = sigmoid(x.dot(theta))
    J = (1./m) * (-y.dot(log(h)) - (1 - y).dot(log(1 - h)))
    return J


def predict(theta, x):
    p_1 = sigmoid(dot(x, theta))
    return p_1 > 0.5


''' Load the data '''
data = loadtxt('ex2data1.txt', delimiter=',')
x = data[:, 0:2]
y = data[:, 2]

''' Plot the data '''
pos = where(y == 1)
neg = where(y == 0)

scatter(x[pos, 0], x[pos, 1], marker='o', c='b')
scatter(x[neg, 0], x[neg, 1], marker='x', c='r')

xlabel('Exam 1 score')
ylabel('Exam 2 score')

legend(['Admitted', 'Not Admitted'])
show()


''' Add x0 to x and initialize theta '''
x = c_[ones(y.size), x]
theta = zeros((3, 1))


''' Optimize the theta using Newton (TNC) algorithm '''
Result = op.minimize(fun=costFunction, x0=theta, args=(x, y), method='TNC', jac=gradient)
opt_theta = Result.x


''' Predict the percentage accuracy of our predictor by feeding it the original dataset '''
g = predict(opt_theta, x)
print(float(y[y == g].size)/float(len(y))*100)
