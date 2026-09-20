# Deep Learning Course – Session 1

## Building Neural Networks with TensorFlow/Keras

Welcome to Session 1 of the Deep Learning course. This session introduces the fundamental building blocks of neural networks using TensorFlow/Keras, including model construction, compilation, training, evaluation, and monitoring with TensorBoard.

---

## Topics Covered

### 1. Environment Setup & Imports
- TensorFlow, PyTorch, and Torchvision version checks
- Keras layers, models, optimizers, losses, metrics, callbacks, and utilities

### 2. Model Building
- `Sequential` model API
- `InputLayer` for explicit input shape definition
- `Dense` fully connected layers
- `Flatten` for converting multi-dimensional inputs to 1D vectors
- CNN building blocks: `Conv2D`, `MaxPooling2D`
- `Dropout` for regularization

### 3. Metrics & Loss Functions
| Loss | Use Case |
|------|----------|
| `MeanSquaredError` | Regression |
| `binary_crossentropy` | Binary classification |
| `categorical_crossentropy` | Multi-class classification (one-hot) |
| `sparse_categorical_crossentropy` | Multi-class classification (integer labels) |

**Key distinction:** Loss is what the model optimizes; metrics are what you monitor.

### 4. Optimizers
- **SGD** – Stochastic Gradient Descent
- **Adam** – Adaptive moment estimation (most commonly used)

### 5. Data Preprocessing
- `to_categorical` – converts class labels to one-hot vectors
- `image_dataset_from_directory` – loads labeled images from folder structure
- `Tokenizer` – converts text into numerical sequences

### 6. Regularization
- `l1_l2` – combined L1/L2 weight penalties
- L1 encourages sparsity; L2 discourages large weights

### 7. Callbacks
- `ModelCheckpoint` – save best model during training
- `EarlyStopping` – stop when validation performance plateaus
- `ReduceLROnPlateau` – lower learning rate when stuck

### 8. Visualization
- `plot_model` – architecture diagram
- Training curves with Matplotlib (accuracy & loss)
- Confusion matrix and classification report with Seaborn/Scikit-learn

---

## Hands-On Examples

### MNIST Classification
- Loaded and reshaped MNIST data (60,000 train / 10,000 test)
- Built a baseline MLP using `sklearn.neural_network.MLPClassifier`
- Built an equivalent model in Keras with `Sequential`
- Trained for 10 epochs with Adam optimizer
- Achieved ~94% test accuracy
- Evaluated with classification report and confusion matrix
- Visualized training curves

### Custom Operations & Layers
Three levels of customization:

| Method | Purpose | Trainable Weights |
|--------|---------|-------------------|
| **Function** | Plain Python operation | No |
| **Lambda layer** | Wrap a function inside the model | No |
| **Custom Layer** | Define your own Keras layer with `build()` and `call()` | Yes |

**Custom layer example:** Implemented a simplified Dense layer using:
```python
class MyCustomLayer(layers.Layer):
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], 1),
                                 initializer='random_normal',
                                 trainable=True)
        self.bias = self.add_weight(shape=(1,),
                                    initializer='random_normal',
                                    trainable=True)

    def call(self, inputs):
        return inputs @ self.W + self.bias
```

Key concepts:
- `inputs` – the tensor entering the layer
- `self.W`, `self.bias` – trainable parameters owned by the layer
- `initializer` – how weights are initialized before training
- `trainable=True` – optimizer is allowed to update the parameter
- `build()` creates weights; `call()` defines what the layer does

### Custom Callbacks
```python
class MyCallBack(callbacks.Callback):
    def on_train_begin(self, logs=None): ...
    def on_train_end(self, logs=None): ...
    def on_epoch_begin(self, epoch, logs=None): ...
    def on_epoch_end(self, epoch, logs=None): ...
```

Callback hierarchy:
```
Training
   └── Epoch
         └── Batch
```

Keras recognizes **exact method names** (`on_train_begin`, `on_epoch_end`, etc.) as hooks. This is a **callback/hook mechanism** – a common programming pattern based on inheritance and method overriding.

### TensorBoard
- Configured `TensorBoard` callback with graph, image, and histogram logging
- Trained with validation data and batch size of 10
- Launched TensorBoard via `!tensorboard --logdir ./logs`

---

## Key Takeaways

1. **Model building** in Keras is modular – stack layers in a `Sequential` model.
2. **Compile** defines the optimizer, loss, and metrics.
3. **Fit** runs the training loop: forward pass → loss → backprop → weight update.
4. **Callbacks** let you hook into any stage of training to monitor or react.
5. **Custom layers** give you full control when built-in layers aren't enough.
6. **TensorBoard** provides visual insight into training dynamics.
7. **Generalization** is managed through dropout, regularization, and early stopping.

---

## What's Next

- What actually happens mathematically during `model.fit()`
- Forward pass → computational graph → backpropagation → gradients → optimizer
- Activation functions, BatchNormalization, GlobalAveragePooling2D
- Embedding, LSTM, GRU, MultiHeadAttention
- GradientTape, `tf.data`, data augmentation, transfer learning
- Transformers, attention, positional encoding, fine-tuning

---

## Requirements

```
tensorflow >= 2.16
torch
torchvision
scikit-learn
seaborn
matplotlib
numpy
```

---

## Notes

- TensorFlow GPU support is not available on native Windows for TF >= 2.11; use WSL2 or TensorFlow-DirectML.
- `tensorflow.keras.utils` is a module of **helper utilities** – not a deep learning concept itself.

---

*Session 1 – Foundations of Deep Learning with TensorFlow/Keras*
