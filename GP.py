import numpy as np
import matplotlib.pyplot as plt


# Parameters
noise = .2
learning_rate = .5

# Create our optimization algorithm (Gradient Descent)
def grad_desc(X, y, num_iters: int):
    X = X.reshape((len(X), 1))
    y = y.reshape((len(y), 1))
    
    # Initialize hyperparameters
    sigma = 1.0
    length_scale = 1.0
    # Log-Marginal Likelihood is baked in as the cost function
    dim_1, dim_2 = rbf(X, X).shape
    for i in range(num_iters):
        if i % 1 == 0:
            print("iteration number: ", i)
            print(sigma, length_scale)
        # Compute the kernel and its inverse
        kernel = rbf(X, X, sigma=sigma, length_scale=length_scale) + np.random.randn(dim_1, dim_2)/10
        inv_kernel = np.linalg.inv(kernel)
        
        # Compute the mean predictions
        mu = np.dot(inv_kernel, y)
        
        # Compute the gradients of the log marginal likelihood
        dL_dsigma = 0.5 * np.trace(np.dot(inv_kernel, np.outer(mu, mu)) - inv_kernel)
        dL_dlength = 0.5 * np.trace(np.dot(inv_kernel, np.outer(mu, mu) * np.power(np.abs(X - X.T), 2) / length_scale**3))
        
        # Update hyperparameters using gradient descent
        sigma -= learning_rate * dL_dsigma
        length_scale -= learning_rate * dL_dlength

    return sigma, length_scale
    

# Create our kernel types
def rbf(data, data2, sigma=1.0, length_scale=1.0):
    # Formula is k(x_i, x_j) = exp(-abs(x_i-x_j)^2/(2*l^2))
    # l is length_scale and is a smoothness parameter. It's a metaparameter that needs tweaking generally
    # This will return a matrix
    return (sigma**2) * np.exp(-.5 * np.power(np.abs(data - data2.T), 2) / np.power(length_scale, 2))

np.random.seed(42)
X = np.sort(5 * np.random.rand(2000, 1), axis=0)
y = np.sin(X) + np.random.normal(0, noise, (X.shape)) # Add in a noise term to better simulate real-world data
vec_pairs = np.hstack([X, y]) # our data matrix
sigma, length_scale = grad_desc(vec_pairs[:,0], vec_pairs[:,1], num_iters=5)

# estimations based on kernel
kernel = rbf(X, X, sigma, length_scale)
inv_kernel = np.linalg.inv(kernel)
mu = np.dot(inv_kernel, y)


plt.scatter(X, y, c='red')
plt.scatter(X, np.sin(X), c='blue')
plt.scatter(X, mu, c='green')
plt.title('Noisy and Noiseless Function Output')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# generate test points
X_test = np.sort(5 * np.random.rand(1000, 1), axis=0)

kernel = rbf(X, X)
K_star = rbf(X, X_test)
inv_kernel = np.linalg.inv(kernel)
mu = np.dot(K_star.T, np.dot(inv_kernel, y))
cov_s = rbf(X_test, X_test) - np.dot(K_star.T, np.dot(inv_kernel, K_star))

X_test.sort()
X_test

plt.scatter(X, y)
plt.scatter(X_test, mu)
plt.show()


