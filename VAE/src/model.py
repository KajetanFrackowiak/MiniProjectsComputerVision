import jax
import jax.numpy as jnp
from flax import nnx


class Encoder(nnx.Module):
    def __init__(self, latents: int, input_size: int, features: list[int], *, rngs: nnx.Rngs):
        self.layers = []
        in_f = 3
        for out_f in features:
            self.layers.append(
                nnx.Conv(in_f, out_f, kernel_size=(3, 3), strides=2, padding="SAME", rngs=rngs)
            )
            in_f = out_f

        self.final_res = input_size // (2 ** len(features))
        self.flattened_dim = (self.final_res**2) * features[-1]

        self.fc_mu = nnx.Linear(self.flattened_dim, latents, rngs=rngs)
        self.fc_logvar = nnx.Linear(self.flattened_dim, latents, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> tuple:
        for layer in self.layers:
            x = nnx.relu(layer(x))  # [B, H, W, C] -> [B, H/2, W/2, features[i]]
        x = x.reshape((x.shape[0], -1))  # [B, H_final, W_final, C_final] -> [B, flattened_dim]
        mu = self.fc_mu(x)  # [B, flattened_dim] -> [B, latents]
        logvar = self.fc_logvar(x)  # [B, flattened_dim] -> [B, latents]
        return mu, logvar


class Decoder(nnx.Module):
    def __init__(self, latents: int, input_size: int, features: list[int], *, rngs: nnx.Rngs):
        self.final_res = input_size // (2 ** len(features))
        self.flattened_dim = (self.final_res**2) * features[-1]

        self.fc = nnx.Linear(latents, self.flattened_dim, rngs=rngs)

        self.layers = []
        in_f = features[-1]
        for out_f in reversed(features[:-1]):
            self.layers.append(
                nnx.ConvTranspose(
                    in_f, out_f, kernel_size=(3, 3), strides=2, padding="SAME", rngs=rngs
                )
            )
            in_f = out_f
        self.layers.append(
            nnx.ConvTranspose(in_f, 3, kernel_size=(3, 3), strides=2, padding="SAME", rngs=rngs)
        )

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        x = nnx.relu(self.fc(z))  # [B, latents] -> [B, flattened_dim]
        x = x.reshape(
            (x.shape[0], self.final_res, self.final_res, -1)
        )  # [B, flattened_dim] -> [B, H_final, W_final, C_final]
        for layer in self.layers[:-1]:
            x = nnx.relu(layer(x))  # [B, H, W, C] -> [B, H*2, W*2, features[i]]
        x = jnp.tanh(self.layers[-1](x))  # [B, H, W, C] -> [B, H*2, W*2, 3]
        return x


class VAE(nnx.Module):
    def __init__(self, latents: int, input_size: int, features: list[int], *, rngs: nnx.Rngs):
        self.encoder = Encoder(latents, input_size, features, rngs=rngs)
        self.decoder = Decoder(latents, input_size, features, rngs=rngs)

    def __call__(self, x: jnp.ndarray, key: jnp.ndarray) -> tuple:
        mu, logvar = self.encoder(x)
        # Reparameterization
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mu.shape)
        z = mu + eps * std

        recon_x = self.decoder(z)
        return recon_x, mu, logvar
