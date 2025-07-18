import numpy as np
from myRegressions import LinearRegression2D
import scipy.stats as stats

# Data
X = np.random.rand(1000, 1)
X = X.flatten()
y = 3 * X + 5 + np.random.randn(1000) * 0.5 

print('start')
# Model
model = LinearRegression2D()
results = model.fit(X, y)
model.print_summary()

# Plot data with regression
# model.plot_regression(X, y)

# Plot loss function value in epochs
# model.plot_loss()


# Comparission
results2 = stats.linregress(X,y)
print("\nscipy.stats.linregress() results:")
print(f"m (slope):     {results2.slope:.4f} ± {results2.stderr:.4f}")
print(f"b (intercept): {results2.intercept:.4f} ± {results2.intercept_stderr:.4f}")
print(f"rvalue = {results2.rvalue:.8f}")