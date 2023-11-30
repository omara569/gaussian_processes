# %%
import numpy as np
import matplotlib.pyplot as plt


# Parameters
noise = .2

# Create our kernel types
def rbf(data, data2, sigma=1.0, length_scale=1.0):
    # Formula is k(x_i, x_j) = exp(-abs(x_i-x_j)^2/(2*l^2))
    # l is length_scale and is a smoothness parameter. It's a metaparameter that needs tweaking generally
    # This will return a matrix
    return (sigma**2) * np.exp(-.5 * np.power(np.abs(data - data2.T), 2) / np.power(length_scale, 2))

np.random.seed(42)
X = np.sort(5 * np.random.rand(2000, 1), axis=0)
y = y = np.sin(X) + np.random.normal(0, noise, (X.shape)) # Add in a noise term to better simulate real-world data
vec_pairs = np.hstack([X, y]) # our data matrix

plt.scatter(X, y, c='red')
plt.scatter(X, np.sin(X), c='blue')
plt.title('Noisy and Noiseless Function Output')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# %%
# generate test points
X_test = np.sort(5 * np.random.rand(1000, 1), axis=0)

# %%
sigma=.5
length_scale=10
kernel = rbf(X, X)
K_star = rbf(X, X_test)
inv_kernel = np.linalg.inv(kernel)
mu = np.dot(K_star.T, np.dot(inv_kernel, y))
cov_s = rbf(X_test, X_test) - np.dot(K_star.T, np.dot(inv_kernel, K_star))


# %%
X_test.sort()
X_test

# %%
plt.scatter(X, y)
plt.scatter(X_test, mu)
plt.show()


