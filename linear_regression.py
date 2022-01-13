def initialize_params(dimension):
    theta_0 = 0
    theta_other = [random.random() for _ in range(dimension)]
    return theta_0, theta_other


def compute_gradient(x, y, theta_0, theta_other, dimension, m):
    gradient_theta_0 = 0
    gradient_theta_other = [0] * dimension

    for i in range(m):
        y_i_hat = sum([x[i][j] * theta_other[j] for j in range(dimension)]) + theta_0
        derror_dy = 2 * (y[i] - y_i_hat)
        for j in range(dimension):
            gradient_theta_other[j] += derror_dy * x[i][j] / n
        gradient_theta_0 += derror_dy / n

    return gradient_theta_0, gradient_theta_other


def update_params(theta_0, theta_other, gradient_theta_0,
    gradient_theta_other, learning_rate):
    theta_0 += gradient_theta_0 * learning_rate
    for i in range(len(theta_1)):
        theta_other[i] += (gradient_theta_other[i] * learning_rate)
    return theta_0, theta_other


def linear_regression(x, y, iterations=100, learning_Rate=0.01):
    n, m = len(x[0]), len(x)
    theta_0, theta_other = initialize_params(n)
    for _ in range(iterations):
        gradient_theta_0, gradient_theta_other = compute_gradient(
            x, y, theta_0, theta_other, n, m)
        theta_0, theta_other = update_params(
            theta_0, theta_other, gradient_theta_0,
            gradient_theta_other, learning_rate)
    return theta_0, theta_other
