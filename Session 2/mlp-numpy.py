import numpy as np
from sklearn.datasets import load_digits


def train_test_split_numpy(X, y, test_size=0.2, random_state=None):
    if random_state:
        np.random.seed(random_state)
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    split_idx = int(X.shape[0] * (1 - test_size))
    return X[indices[:split_idx]], X[indices[split_idx:]], y[indices[:split_idx]], y[indices[split_idx:]]


def standard_scaler_numpy(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std


# Load dataset
digits = load_digits()
X = digits.data / 16.0  # Normalize pixel values
y = digits.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split_numpy(X, y, test_size=0.2, random_state=42)
X_train, X_test = standard_scaler_numpy(X_train, X_test)


# Activation functions
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def cross_entropy(y_pred, y_true):
    n_samples = y_true.shape[0]
    log_likelihood = -np.log(y_pred[range(n_samples), y_true] + 1e-9)
    return np.sum(log_likelihood) / n_samples


def l2_regularization(params, reg_lambda):
    return 0.5 * reg_lambda * sum(np.sum(w ** 2) for k, w in params.items() if 'W' in k)


def adam_update(params, grads, v, s, t, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    for key in params:
        v[key] = beta1 * v[key] + (1 - beta1) * grads[key]
        s[key] = beta2 * s[key] + (1 - beta2) * (grads[key] ** 2)

        v_corrected = v[key] / (1 - beta1 ** t)
        s_corrected = s[key] / (1 - beta2 ** t)

        params[key] -= lr * v_corrected / (np.sqrt(s_corrected) + epsilon)
    return params, v, s


class MLPClassifier:
    def __init__(self, input_dim, hidden_dim, output_dim, lr=0.001, reg_lambda=0.01):
        self.lr = lr
        self.reg_lambda = reg_lambda
        self.params = {
            'W1': np.random.randn(input_dim, hidden_dim) * 0.01,
            'b1': np.zeros((1, hidden_dim)),
            'W2': np.random.randn(hidden_dim, output_dim) * 0.01,
            'b2': np.zeros((1, output_dim))
        }
        self.v = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.s = {key: np.zeros_like(value) for key, value in self.params.items()}
        self.t = 1

    def forward(self, X):
        self.z1 = X @ self.params['W1'] + self.params['b1']
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.params['W2'] + self.params['b2']
        self.a2 = softmax(self.z2)
        return self.a2

    def backward(self, X, y_true):
        m = X.shape[0]
        y_pred = self.a2
        y_true_one_hot = np.zeros_like(y_pred)
        y_true_one_hot[np.arange(m), y_true] = 1
        dz2 = y_pred - y_true_one_hot
        dW2 = self.a1.T @ dz2 / m + self.reg_lambda * self.params['W2']
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        dz1 = dz2 @ self.params['W2'].T * relu_derivative(self.z1)
        dW1 = X.T @ dz1 / m + self.reg_lambda * self.params['W1']
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, verbose=True, early_stop=5,
              loss_threshold=1e-4):
        best_val_loss = float('inf')
        patience = 0
        for epoch in range(epochs):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train, y_train = X_train[indices], y_train[indices]

            for i in range(0, X_train.shape[0], batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                y_pred = self.forward(X_batch)
                grads = self.backward(X_batch, y_batch)
                self.params, self.v, self.s = adam_update(self.params, grads, self.v, self.s, self.t, self.lr)
                self.t += 1

            train_loss = cross_entropy(self.forward(X_train), y_train) + l2_regularization(self.params, self.reg_lambda)
            val_loss = cross_entropy(self.forward(X_val), y_val) + l2_regularization(self.params, self.reg_lambda)
            val_acc = np.mean(self.predict(X_val) == y_val)

            if verbose:
                print(
                    f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            if early_stop is not None:
                if best_val_loss - val_loss < loss_threshold:
                    patience += 1
                else:
                    best_val_loss = val_loss
                    patience = 0

                if patience >= early_stop:
                    print("Early stopping triggered!")
                    break

    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)


X_train, X_val, y_train, y_val = train_test_split_numpy(X_train, y_train, test_size=0.2, random_state=42)
mlp = MLPClassifier(input_dim=64, hidden_dim=128, output_dim=10, lr=0.001, reg_lambda=0.001)
mlp.train(X_train, y_train, X_val, y_val, epochs=100, batch_size=64, verbose=True, early_stop=5, loss_threshold=1e-4)
y_pred = mlp.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f'Test Accuracy: {accuracy:.4f}')
