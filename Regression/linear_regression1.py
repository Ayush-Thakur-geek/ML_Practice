import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
# Example data
x = np.random.rand(100, 1)  # 100 samples, 1 feature
y = np.random.rand(100, 1)  # 100 target values

random_indices = np.random.permutation(100)

# Training set
x_train = x[random_indices[:70]]
y_train = y[random_indices[:70]]

# Validation set
x_val = x[random_indices[70:85]]
y_val = y[random_indices[70:85]]

#Test set
x_test = x[random_indices[85:]]
y_test = y[random_indices[85:]]

model = LinearRegression()

x_train_for_line_fitting = np.asarray(x_train.reshape(len(x_train), 1))
y_train_for_line_fitting = np.asarray(y_train.reshape(len(y_train), 1))

model.fit(x_train_for_line_fitting, y_train_for_line_fitting)

# Generate predictions
y_pred = model.predict(x_train_for_line_fitting)

# Plotting
plt.scatter(x_train, y_train, color='blue', label='Original data')
plt.plot(x_train, y_pred, color='red', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression')
plt.legend()
plt.show()