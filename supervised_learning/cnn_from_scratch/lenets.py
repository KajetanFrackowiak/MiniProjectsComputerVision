from cnn import Conv2d, MaxPool2d, Flatten, Linear, Adam, ReLU
import numpy as np
import pickle
import os


class LeNet5:
    def __init__(self):
        self.conv1 = Conv2d(1, 6, 5, stride=1, padding=2)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(2, stride=2)
        self.conv2 = Conv2d(6, 16, 5, stride=1, padding=0)
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(2, stride=2)
        self.conv3 = Conv2d(16, 120, 5, stride=1, padding=0)
        self.relu3 = ReLU()
        self.flatten = Flatten()
        self.fc1 = Linear(120, 84)
        self.relu4 = ReLU()
        self.fc2 = Linear(84, 10)

        # Create optimizer
        self.optimizer = Adam([self.conv1, self.conv2, self.conv3, self.fc1, self.fc2])

    def forward(self, x):
        # Store all intermediate values for backward pass
        self.layers = []

        x = self.conv1.forward(x)
        self.layers.append(self.conv1)
        x = self.relu1.forward(x)
        self.layers.append(self.relu1)
        x = self.pool1.forward(x)
        self.layers.append(self.pool1)

        x = self.conv2.forward(x)
        self.layers.append(self.conv2)
        x = self.relu2.forward(x)
        self.layers.append(self.relu2)
        x = self.pool2.forward(x)
        self.layers.append(self.pool2)

        x = self.conv3.forward(x)
        self.layers.append(self.conv3)
        x = self.relu3.forward(x)
        self.layers.append(self.relu3)

        x = self.flatten.forward(x)
        self.layers.append(self.flatten)
        x = self.fc1.forward(x)
        self.layers.append(self.fc1)
        x = self.relu4.forward(x)
        self.layers.append(self.relu4)
        x = self.fc2.forward(x)
        self.layers.append(self.fc2)

        return x

    def softmax(self, x):
        # Stable softmax implementation
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def cross_entropy_loss(self, preds, targets):
        # Apply softmax
        softmax_preds = self.softmax(preds)

        # Clip predictions to prevent log(0)
        softmax_preds = np.clip(softmax_preds, 1e-15, 1 - 1e-15)

        # One-hot encode targets
        batch_size = targets.shape[0]
        targets_one_hot = np.zeros((batch_size, 10))
        targets_one_hot[np.arange(batch_size), targets] = 1

        # Calculate cross-entropy loss
        loss = -np.mean(np.sum(targets_one_hot * np.log(softmax_preds), axis=1))

        # Calculate gradient (softmax derivative with cross-entropy)
        grad = (softmax_preds - targets_one_hot) / batch_size

        return loss, grad

    def backward(self, grad_output):
        # Backpropagate through all layers in reverse order
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)

    def train_step(self, x, y):
        # Forward pass
        logits = self.forward(x)

        # Calculate loss and gradients
        loss, grad_output = self.cross_entropy_loss(logits, y)

        # Backward pass
        self.backward(grad_output)

        # Update weights
        self.optimizer.step()

        return loss

    def save_model(self, filepath):
        """Save the model weights and architecture to a file."""
        model_data = {
            "conv1_weights": self.conv1.weights,
            "conv1_bias": self.conv1.bias,
            "conv2_weights": self.conv2.weights,
            "conv2_bias": self.conv2.bias,
            "conv3_weights": self.conv3.weights,
            "conv3_bias": self.conv3.bias,
            "fc1_weights": self.fc1.weights,
            "fc1_bias": self.fc1.bias,
            "fc2_weights": self.fc2.weights,
            "fc2_bias": self.fc2.bias,
        }

        # Create directory if it doesn't exist
        os.makedirs(
            os.path.dirname(filepath) if os.path.dirname(filepath) else ".",
            exist_ok=True,
        )

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model weights from a file."""
        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        self.conv1.weights = model_data["conv1_weights"]
        self.conv1.bias = model_data["conv1_bias"]
        self.conv2.weights = model_data["conv2_weights"]
        self.conv2.bias = model_data["conv2_bias"]
        self.conv3.weights = model_data["conv3_weights"]
        self.conv3.bias = model_data["conv3_bias"]
        self.fc1.weights = model_data["fc1_weights"]
        self.fc1.bias = model_data["fc1_bias"]
        self.fc2.weights = model_data["fc2_weights"]
        self.fc2.bias = model_data["fc2_bias"]
        print(f"Model loaded from {filepath}")

    def predict(self, x):
        """Make predictions without storing gradients (for inference)."""
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)

        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)

        x = self.conv3.forward(x)
        x = self.relu3.forward(x)

        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu4.forward(x)
        x = self.fc2.forward(x)

        return self.softmax(x)
