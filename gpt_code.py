# The following is from ChatGPT and shows the basics of Gaussian Processes Using SK Learn:
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Generate some synthetic data
np.random.seed(42)
X = np.sort(5 * np.random.rand(20, 1), axis=0)
y = np.sin(X).ravel()

# Define the kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

# Create Gaussian Process Regressor
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make predictions on new data points
X_new = np.linspace(0, 5, 1000)[:, np.newaxis]
y_pred, sigma = gp.predict(X_new, return_std=True)

# Plot the results
plt.figure(figsize=(8, 4))
plt.scatter(X, y, c='r', s=20, zorder=10, edgecolors=(0, 0, 0))
plt.plot(X_new, y_pred, 'k', label='Prediction')
plt.fill_between(X_new.ravel(), y_pred - sigma, y_pred + sigma, alpha=0.2, color='k')
plt.title('Gaussian Process Regression')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()