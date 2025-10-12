import jax
import jax.numpy as jnp
from flax import nnx


class SpectralNormConv(nnx.Module):
    """Conv layer with spectral normalization for NHWC format."""

    def __init__(
        self,
        out_features: int,
        kernel_size: tuple,
        padding: str = "SAME",
        strides: int = 1,
        n_power_iterations: int = 1,
        *,
        rngs: nnx.Rngs,
    ):
        self.out_features = out_features
        self.kernel_size = kernel_size
        self.padding = padding
        self.strides = strides
        self.n_power_iterations = n_power_iterations

        # Note: We'll initialize kernel and u in the first forward pass
        # since we need to know the input channels
        self.kernel = None
        self.bias = None
        self.u = None
        self.rngs = rngs

    def __call__(self, x):
        # Lazy initialization on first call
        if self.kernel is None:
            in_channels = x.shape[-1]
            kernel_shape = self.kernel_size + (in_channels, self.out_features)
            self.kernel = nnx.Param(
                nnx.initializers.lecun_normal()(self.rngs(), kernel_shape)
            )
            self.bias = nnx.Param(jnp.zeros((self.out_features,)))

            # Initialize u vector for power iteration
            w_reshaped = self.kernel.value.reshape(-1, self.kernel.value.shape[-1])
            self.u = nnx.Variable(
                jax.random.normal(self.rngs(), (1, w_reshaped.shape[-1]))
            )

        # Flatten kernel for spectral norm computation
        w_reshaped = self.kernel.value.reshape(-1, self.kernel.value.shape[-1])

        # Power iteration to estimate spectral norm
        u_val = self.u.value
        for _ in range(self.n_power_iterations):
            v = jnp.dot(u_val, w_reshaped.T)
            v = v / (jnp.linalg.norm(v) + 1e-12)
            u_val = jnp.dot(v, w_reshaped)
            u_val = u_val / (jnp.linalg.norm(u_val) + 1e-12)

        # Update u for next iteration
        self.u.value = u_val

        # Compute spectral norm (largest singular value)
        sigma = jnp.dot(jnp.dot(v, w_reshaped), u_val.T)

        # Normalize the kernel
        kernel_normalized = self.kernel.value / (sigma + 1e-12)

        # Apply convolution with normalized kernel
        output = (
            jax.lax.conv_general_dilated(
                x,
                kernel_normalized,
                window_strides=(
                    self.strides
                    if isinstance(self.strides, tuple)
                    else (self.strides, self.strides)
                ),
                padding=self.padding,
                dimension_numbers=("NHWC", "HWIO", "NHWC"),
            )
            + self.bias.value
        )

        return output


class SpectralNormLinear(nnx.Module):
    """Linear/Dense layer with spectral normalization."""

    def __init__(
        self, out_features: int, n_power_iterations: int = 1, *, rngs: nnx.Rngs
    ):
        self.out_features = out_features
        self.n_power_iterations = n_power_iterations

        # Lazy initialization
        self.kernel = None
        self.bias = None
        self.u = None
        self.rngs = rngs

    def __call__(self, x):
        # Lazy initialization on first call
        if self.kernel is None:
            in_features = x.shape[-1]
            self.kernel = nnx.Param(
                nnx.initializers.lecun_normal()(
                    self.rngs(), (in_features, self.out_features)
                )
            )
            self.bias = nnx.Param(jnp.zeros((self.out_features,)))
            self.u = nnx.Variable(
                jax.random.normal(self.rngs(), (1, self.out_features))
            )

        # Power iteration to estimate spectral norm
        u_val = self.u.value
        for _ in range(self.n_power_iterations):
            v = jnp.dot(u_val, self.kernel.value.T)
            v = v / (jnp.linalg.norm(v) + 1e-12)
            u_val = jnp.dot(v, self.kernel.value)
            u_val = u_val / (jnp.linalg.norm(u_val) + 1e-12)

        # Update u for next iteration
        self.u.value = u_val

        # Compute spectral norm
        sigma = jnp.dot(jnp.dot(v, self.kernel.value), u_val.T)

        # Normalize the kernel
        kernel_normalized = self.kernel.value / (sigma + 1e-12)

        # Apply linear layer with normalized kernel
        return jnp.dot(x, kernel_normalized) + self.bias.value
