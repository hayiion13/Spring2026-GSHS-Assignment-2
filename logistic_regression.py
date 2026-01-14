import numpy as np
from sklearn.linear_model import LogisticRegression

def logistic_regression(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray) -> np.ndarray:
    learning_rate = 0.1
    num_iterations = 1000

    n_samples, n_features = x_train.shape

    weights = np.zeros(n_features)
    bias = 0.0

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    for _ in range(num_iterations):
        linear_model = np.dot(x_train, weights) + bias
        y_pred = sigmoid(linear_model)

        dw = (1 / n_samples) * np.dot(x_train.T, (y_pred - y_train))
        db = (1 / n_samples) * np.sum(y_pred - y_train)

        weights -= learning_rate * dw
        bias -= learning_rate * db

    linear_model_test = np.dot(x_test, weights) + bias
    y_test_pred = sigmoid(linear_model_test)

    y_pred_labels = (y_test_pred >= 0.5).astype(int)

    return y_pred_labels
