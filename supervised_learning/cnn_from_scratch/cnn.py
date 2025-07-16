import numpy as np


class Conv2d:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize weights with He initialization
        self.weights = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.bias = np.zeros(out_channels)

    def forward(self, x):
        batch_size, in_channels, height, width = x.shape
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Apply padding to input
        if self.padding > 0:
            x_padded = np.pad(
                x,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
            )
        else:
            x_padded = x
        self.x_padded = x_padded

        # Create output tensor
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))

        # Process each output position efficiently
        for i in range(out_height):
            i_max = i * self.stride + self.kernel_size
            for j in range(out_width):
                j_max = j * self.stride + self.kernel_size

                # Extract patches for all batches and input channels at once
                patches = x_padded[
                    :, :, i * self.stride : i_max, j * self.stride : j_max
                ]

                # Reshape patches to (batch_size, in_channels * kernel_size * kernel_size)
                patches_flat = patches.reshape(batch_size, -1)

                # Reshape weights to (out_channels, in_channels * kernel_size * kernel_size)
                weights_flat = self.weights.reshape(self.out_channels, -1)

                # Matrix multiplication
                conv_result = np.dot(weights_flat, patches_flat.T)

                # Store result
                output[:, :, i, j] = conv_result.T

        # Add bias
        output += self.bias.reshape(1, -1, 1, 1)

        return output

    def backward(self, grad_output):
        batch_size, out_channels, out_height, out_width = grad_output.shape

        # Initialize gradients
        self.weights_grad = np.zeros_like(self.weights)
        self.bias_grad = np.sum(grad_output, axis=(0, 2, 3))

        # Initialize gradient for input
        grad_input_padded = np.zeros_like(self.x_padded)

        # Efficient backward pass using vectorized operations
        weights_flat = self.weights.reshape(self.out_channels, -1)

        for i in range(out_height):
            i_max = i * self.stride + self.kernel_size
            for j in range(out_width):
                j_max = j * self.stride + self.kernel_size

                # Get gradients for this position across all batches and output channels
                grad_pos = grad_output[:, :, i, j]  # (batch_size, out_channels)

                # Get input patches for this position
                input_patches = self.x_padded[
                    :, :, i * self.stride : i_max, j * self.stride : j_max
                ]
                input_patches_flat = input_patches.reshape(batch_size, -1)

                # Update weight gradients efficiently
                weight_grad_contrib = np.dot(grad_pos.T, input_patches_flat)
                self.weights_grad += weight_grad_contrib.reshape(self.weights.shape)

                # Update input gradients efficiently
                input_grad_contrib = np.dot(grad_pos, weights_flat)
                input_grad_contrib = input_grad_contrib.reshape(
                    batch_size, self.in_channels, self.kernel_size, self.kernel_size
                )

                grad_input_padded[
                    :, :, i * self.stride : i_max, j * self.stride : j_max
                ] += input_grad_contrib

        # Remove padding from grad_input if needed
        if self.padding > 0:
            grad_input = grad_input_padded[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ]
        else:
            grad_input = grad_input_padded

        return grad_input


class MaxPool2d:
    def __init__(self, kernel_size, stride=None, padding=0):
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        # Apply padding
        if self.padding > 0:
            x_padded = np.pad(
                x,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
            )
        else:
            x_padded = x
        self.x_padded = x_padded

        output = np.zeros((batch_size, channels, out_height, out_width))

        # Store max positions for backward pass
        self.pooling_cache = []

        # Optimized pooling - vectorized operations where possible
        for i in range(out_height):
            for j in range(out_width):
                h_start = i * self.stride
                h_end = h_start + self.kernel_size
                w_start = j * self.stride
                w_end = w_start + self.kernel_size

                # Extract window for all batches and channels simultaneously
                window = x_padded[:, :, h_start:h_end, w_start:w_end]

                # Reshape and find max
                window_flat = window.reshape(batch_size, channels, -1)
                max_vals = np.max(window_flat, axis=2)
                max_indices = np.argmax(window_flat, axis=2)

                output[:, :, i, j] = max_vals

                # Store cache for backward pass
                self.pooling_cache.append((h_start, w_start, max_indices))

        return output

    def backward(self, grad_output):
        batch_size, channels, out_height, out_width = grad_output.shape
        grad_input = np.zeros_like(self.x_padded)

        # Use cached pooling information
        cache_idx = 0
        for i in range(out_height):
            for j in range(out_width):
                h_start, w_start, max_indices = self.pooling_cache[cache_idx]
                cache_idx += 1

                # Convert flat indices back to 2D coordinates
                max_h_offset = max_indices // self.kernel_size
                max_w_offset = max_indices % self.kernel_size

                # Add gradients to the max positions
                for b in range(batch_size):
                    for c in range(channels):
                        max_h = h_start + max_h_offset[b, c]
                        max_w = w_start + max_w_offset[b, c]
                        grad_input[b, c, max_h, max_w] += grad_output[b, c, i, j]

        # Remove padding from grad_input if needed
        if self.padding > 0:
            grad_input = grad_input[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ]
        return grad_input


class ReLU:
    def forward(self, x):
        self.input = x  # Store for backward
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * (self.input > 0)


class Flatten:
    def forward(self, x):
        self.input_shape = x.shape  # Store original shape for backward
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output):
        return grad_output.reshape(self.input_shape)


class Linear:
    def __init__(self, in_features, out_features):
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights with He initialization
        self.weights = np.random.randn(in_features, out_features) * np.sqrt(
            2.0 / in_features
        )
        self.bias = np.zeros(out_features)

    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias

    def backward(self, grad_output):
        grad_input = np.dot(grad_output, self.weights.T)
        self.weights_grad = np.dot(self.input.T, grad_output)
        self.bias_grad = np.sum(grad_output, axis=0)
        return grad_input


class Adam:
    def __init__(self, layers, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = layers  # List of layers with weights and gradients
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_weights = {}
        self.v_weights = {}
        self.m_bias = {}
        self.v_bias = {}
        self.t = 0

        # Initialize momentum for each layer
        for i, layer in enumerate(layers):
            if hasattr(layer, "weights"):
                self.m_weights[i] = np.zeros_like(layer.weights)
                self.v_weights[i] = np.zeros_like(layer.weights)
                self.m_bias[i] = np.zeros_like(layer.bias)
                self.v_bias[i] = np.zeros_like(layer.bias)

    def step(self):
        self.t += 1
        bias_correction1 = 1 - self.beta1**self.t
        bias_correction2 = 1 - self.beta2**self.t

        for i, layer in enumerate(self.layers):
            if hasattr(layer, "weights") and hasattr(layer, "weights_grad"):
                # Update weights with vectorized operations
                self.m_weights[i] = (
                    self.beta1 * self.m_weights[i]
                    + (1 - self.beta1) * layer.weights_grad
                )
                self.v_weights[i] = self.beta2 * self.v_weights[i] + (
                    1 - self.beta2
                ) * (layer.weights_grad**2)

                m_hat_w = self.m_weights[i] / bias_correction1
                v_hat_w = self.v_weights[i] / bias_correction2
                layer.weights -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)

                # Update bias with vectorized operations
                self.m_bias[i] = (
                    self.beta1 * self.m_bias[i] + (1 - self.beta1) * layer.bias_grad
                )
                self.v_bias[i] = self.beta2 * self.v_bias[i] + (1 - self.beta2) * (
                    layer.bias_grad**2
                )

                m_hat_b = self.m_bias[i] / bias_correction1
                v_hat_b = self.v_bias[i] / bias_correction2
                layer.bias -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)
