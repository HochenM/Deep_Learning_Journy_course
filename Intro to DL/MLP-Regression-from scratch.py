# For Regression
import numpy as np
from sklearn.datasets import load_diabetes


def train_test_split_numpy(X, y, test_size=0.2, random_state=None):
    if random_state is not None:
        np.random.seed(random_state)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    split_idx = int(X.shape[0] * (1 - test_size))

    return (
        X[indices[:split_idx]],
        X[indices[split_idx:]],
        y[indices[:split_idx]],
        y[indices[split_idx:]]
    )


def standard_scaler_numpy(X_train, X_test):
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-8

    return (
        (X_train - mean) / std,
        (X_test - mean) / std
    )


# Load dataset

diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target


# Split dataset

X_train, X_test, y_train, y_test = train_test_split_numpy(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train, X_test = standard_scaler_numpy(X_train, X_test)


# Activation functions

def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return (x > 0).astype(float)


# MSE Loss

def mse_loss(y_pred, y_true):
    y_true = y_true.reshape(-1, 1)
    return np.mean((y_pred - y_true) ** 2)


def l2_regularization(params, reg_lambda):
    return 0.5 * reg_lambda * sum(
        np.sum(w ** 2)
        for k, w in params.items()
        if 'W' in k
    )


def adam_update(
    params,
    grads,
    v,
    s,
    t,
    lr=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8
):

    for key in params:

        v[key] = (
            beta1 * v[key]
            + (1 - beta1) * grads[key]
        )

        s[key] = (
            beta2 * s[key]
            + (1 - beta2) * (grads[key] ** 2)
        )

        v_corrected = v[key] / (1 - beta1 ** t)
        s_corrected = s[key] / (1 - beta2 ** t)

        params[key] -= (
            lr
            * v_corrected
            / (np.sqrt(s_corrected) + epsilon)
        )

    return params, v, s


class MLPRegressor:

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        lr=0.001,
        reg_lambda=0.01
    ):

        self.lr = lr
        self.reg_lambda = reg_lambda

        self.params = {
            'W1': np.random.randn(input_dim, hidden_dim) * 0.01,
            'b1': np.zeros((1, hidden_dim)),
            'W2': np.random.randn(hidden_dim, output_dim) * 0.01,
            'b2': np.zeros((1, output_dim))
        }

        self.v = {
            key: np.zeros_like(value)
            for key, value in self.params.items()
        }

        self.s = {
            key: np.zeros_like(value)
            for key, value in self.params.items()
        }

        self.t = 1


    def forward(self, X):

        self.z1 = X @ self.params['W1'] + self.params['b1']
        self.a1 = relu(self.z1)

        self.z2 = self.a1 @ self.params['W2'] + self.params['b2']

        # Linear output for regression
        self.a2 = self.z2

        return self.a2


    def backward(self, X, y_true):

        m = X.shape[0]

        y_true = y_true.reshape(-1, 1)
        y_pred = self.a2

        # MSE + linear output
        dz2 = 2 * (y_pred - y_true) / m

        dW2 = (
            self.a1.T @ dz2
            + self.reg_lambda * self.params['W2']
        )

        db2 = np.sum(
            dz2,
            axis=0,
            keepdims=True
        )

        dA1 = dz2 @ self.params['W2'].T

        dz1 = (
            dA1
            * relu_derivative(self.z1)
        )

        dW1 = (
            X.T @ dz1
            + self.reg_lambda * self.params['W1']
        )

        db1 = np.sum(
            dz1,
            axis=0,
            keepdims=True
        )

        return {
            'W1': dW1,
            'b1': db1,
            'W2': dW2,
            'b2': db2
        }


    def train(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
        epochs=50,
        batch_size=64,
        verbose=True,
        early_stop=5,
        loss_threshold=1e-4
    ):

        best_val_loss = float('inf')
        patience = 0

        for epoch in range(epochs):

            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)

            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(
                0,
                X_train.shape[0],
                batch_size
            ):

                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                y_pred = self.forward(X_batch)

                grads = self.backward(
                    X_batch,
                    y_batch
                )

                self.params, self.v, self.s = adam_update(
                    self.params,
                    grads,
                    self.v,
                    self.s,
                    self.t,
                    self.lr
                )

                self.t += 1


            train_loss = (
                mse_loss(
                    self.forward(X_train),
                    y_train
                )
                + l2_regularization(
                    self.params,
                    self.reg_lambda
                )
            )

            val_loss = (
                mse_loss(
                    self.forward(X_val),
                    y_val
                )
                + l2_regularization(
                    self.params,
                    self.reg_lambda
                )
            )


            if verbose:
                print(
                    f'Epoch {epoch + 1}, '
                    f'Train Loss: {train_loss:.4f}, '
                    f'Val Loss: {val_loss:.4f}'
                )


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
        return self.forward(X)


# Validation split

X_train, X_val, y_train, y_val = train_test_split_numpy(
    X_train,
    y_train,
    test_size=0.2,
    random_state=42
)


# Model

mlp = MLPRegressor(
    input_dim=10,
    hidden_dim=128,
    output_dim=1,
    lr=0.001,
    reg_lambda=0.001
)


# Train

mlp.train(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=100,
    batch_size=64,
    verbose=True,
    early_stop=5,
    loss_threshold=1e-4
)


# Prediction

y_pred = mlp.predict(X_test)


# Test MSE

test_mse = mse_loss(y_pred, y_test)

print(f'Test MSE: {test_mse:.4f}')