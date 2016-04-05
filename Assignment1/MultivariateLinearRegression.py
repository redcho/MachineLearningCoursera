from numpy import loadtxt, zeros, ones, array, mean, std, arange, dot
from numpy.linalg import inv
from pylab import show,  xlabel, ylabel, plot

def feature_normalize(X):
    '''
    Returns a normalized version of X where
    the mean value of each feature is 0 and the standard deviation
    is 1. This is often a good preprocessing step to do when
    working with learning algorithms.
    '''

    mean_r = []
    std_r = []

    X_norm = X.copy()

    n_c = X.shape[1]
    for i in range(n_c):
        m = mean(X[:, i])
        s = std(X[:, i])
        mean_r.append(m)
        std_r.append(s)
        X_norm[:, i] = (X_norm[:, i] - m) / s

    return X_norm, mean_r, std_r


def compute_cost(X, y, theta):
    '''
    Comput cost for linear regression
    '''
    #Number of training samples
    m = y.size

    predictions = X.dot(theta).flatten()

    sqErrors = (predictions - y)

    J = (1.0 / (2 * m)) * sqErrors.T.dot(sqErrors)

    return J


def gradient_descent(X, y, theta, alpha, num_iters):
    '''
    Performs gradient descent to learn theta
    by taking num_items gradient steps with learning
    rate alpha
    '''
    m = y.size
    J_history = zeros(shape=(num_iters, 1))
    for i in range(num_iters):
        temptheta = zeros(shape=(3, 1))
        for it in range(theta.size):
            temp = X[:, it]
            predictions = X.dot(theta).flatten()
            errors_x1 = (predictions - y) * temp
            temptheta[it][0] = alpha * (1.0 / m) * errors_x1.sum()
        theta -= temptheta
        J_history[i, 0] = compute_cost(X, y, theta)
    return theta, J_history


''' Initial Data '''
data = loadtxt('ex1data2.txt', delimiter=',')
x = data[:, :2]
y = data[:, 2]
m = y.size


''' Normalize '''
x_norm, mean_r, std_r = feature_normalize(x)


''' Add x0 to x '''
it = ones(shape=(m, 3))
it[:, 1:3] = x_norm


''' Initialize variables '''
iterations = 50
alpha = 0.108
theta = zeros(shape=(3, 1))


''' Gradient Descent '''
theta, J_history = gradient_descent(it, y, theta, alpha, iterations)


''' Plot history of J values '''
plot(arange(iterations), J_history)
xlabel('Iterations')
ylabel('Cost Function')
show()


''' Prediction '''
price = array([1.0,   ((1650.0 - mean_r[0]) / std_r[0]), ((3 - mean_r[1]) / std_r[1])]).dot(theta)
print('Predicted price of a 1650 sq-ft, 3 br house: %f', price)


''' Normal Equation '''
bigx = ones(shape=(m, 3))
bigx[:, 1:3] = data[:, :2]

z = inv(dot(bigx.transpose(), bigx))
last = dot(dot(z, bigx.transpose()), y)
price = array([1.0, 1650.0, 3]).dot(last)
print(price)
