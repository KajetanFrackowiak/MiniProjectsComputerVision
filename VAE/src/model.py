import jax
import jax.numpy as jnp
from flax import nnx


class Encoder(nnx.Module):
    def __init__(self, latents: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(3, 32, kernel_size=(3, 3), strides=2, padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), strides=2, padding="SAME", rngs=rngs)
        self.conv3 = nnx.Conv(64, 128, kernel_size=(3, 3), strides=2, padding="SAME", rngs=rngs)

        self.fc_mu = nnx.Linear(2048, latents, rngs=rngs)
        self.fc_logvar = nnx.Linear(2048, latents, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> tuple:
        x = nnx.relu(self.conv1(x))  # [B, 16, 16, 32]
        x = nnx.relu(self.conv2(x))  # [B, 8, 8, 64]
        x = nnx.relu(self.conv3(x))  # [B, 4, 4, 128]
        x = x.reshape((x.shape[0], -1))  # [B, 2048]
        return self.fc_mu(x), self.fc_logvar(x)


class Decoder(nnx.Module):
    def __init__(self, latents: int, *, rngs: nnx.Rngs):
        self.fc = nnx.Linear(latents, 2048, rngs=rngs)
        self.deconv1 = nnx.ConvTranspose(
            128, 64, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs
        )
        self.deconv2 = nnx.ConvTranspose(
            64, 32, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs
        )
        self.deconv3 = nnx.ConvTranspose(
            32, 3, kernel_size=(3, 3), strides=(2, 2), padding="SAME", rngs=rngs
        )

    def __call__(self, z: jnp.ndarray) -> jnp.ndarray:
        x = nnx.relu(self.fc(z))  # [B, 2048]
        x = x.reshape((-1, 4, 4, 128))  # [B, 4, 4, 128]
        x = nnx.relu(self.deconv1(x))  # [B, 8, 8, 64]
        x = nnx.relu(self.deconv2(x))  # [B, 16, 16, 32]
        x = jax.nn.sigmoid(self.deconv3(x))  # [B, 32, 32, 3]
        return x


class VAE(nnx.Module):
    def __init__(self, latents: int, *, rngs: nnx.Rngs):
        self.encoder = Encoder(latents, rngs=rngs)
        self.decoder = Decoder(latents, rngs=rngs)

    def __call__(self, x: jnp.ndarray, key: jnp.ndarray) -> tuple:
        mu, logvar = self.encoder(x)
        # Reparameterization
        std = jnp.exp(0.5 * logvar)
        eps = jax.random.normal(key, mu.shape)
        z = mu + eps * std

        recon_x = self.decoder(z)
        return recon_x, mu, logvar
