import jax.numpy as jnp
from flax import nnx
import jax
import wandb
import os
from tqdm import tqdm

from losses import wgan_loss, wgan_gp_loss, gradient_penalty


@nnx.jit(static_argnums=(5,))  # gan_type is static (arg index 5)
def train_discriminator_step_jit(
    critic, optimizer, generator, batch_img, fake, gan_type, gp_rng
):
    def d_loss_fn(critic):
        D_real = critic(batch_img)
        D_fake = critic(fake)

        if gan_type == "wgan":
            d_loss, _ = wgan_loss(D_real, D_fake)
        else:
            gp = gradient_penalty(critic, batch_img, fake, gp_rng)
            d_loss, _ = wgan_gp_loss(D_real, D_fake, gp)

        return d_loss

    grads = nnx.grad(d_loss_fn)(critic)
    optimizer.update(grads)

    # Compute final loss for logging
    D_real = critic(batch_img)
    D_fake = critic(fake)
    if gan_type == "wgan":
        d_loss, _ = wgan_loss(D_real, D_fake)
    else:
        gp = gradient_penalty(critic, batch_img, fake, gp_rng)
        d_loss, _ = wgan_gp_loss(D_real, D_fake, gp)

    return d_loss


@nnx.jit(static_argnums=(4,))  # gan_type is static (arg index 4)
def train_generator_step_jit(generator, optimizer, critic, z, gan_type):
    def g_loss_fn(generator):
        fake = generator(z)
        D_fake = critic(fake)

        if gan_type == "wgan":
            _, g_loss = wgan_loss(jnp.zeros_like(D_fake), D_fake)
        else:
            _, g_loss = wgan_gp_loss(jnp.zeros_like(D_fake), D_fake, 0.0)

        return g_loss

    grads = nnx.grad(g_loss_fn)(generator)
    optimizer.update(grads)

    # Compute final loss for logging
    fake = generator(z)
    D_fake = critic(fake)
    if gan_type == "wgan":
        _, g_loss = wgan_loss(jnp.zeros_like(D_fake), D_fake)
    else:
        _, g_loss = wgan_gp_loss(jnp.zeros_like(D_fake), D_fake, 0.0)

    return g_loss


class GANTrainer:
    def __init__(
        self,
        generator,
        critic,
        generator_tx,
        critic_tx,
        gan_type,
        latent_dim,
        critic_steps,
    ):
        self.generator = generator
        self.critic = critic
        self.gan_type = gan_type
        self.latent_dim = latent_dim
        self.critic_steps = critic_steps
        self.losses = {"critic_loss": [], "generator_loss": []}
        self.current_d_loss = 0.0
        self.current_g_loss = 0.0

        self.G_tx = generator_tx
        self.D_tx = critic_tx
        self.G_optimizer = None
        self.D_optimizer = None

    def train_discriminator_step(self, batch, rng):
        z_rng, gp_rng = jax.random.split(rng)
        z = jax.random.normal(z_rng, (batch["img"].shape[0], self.latent_dim))

        fake = jax.lax.stop_gradient(self.generator(z))

        d_loss = train_discriminator_step_jit(
            self.critic,
            self.D_optimizer,
            self.generator,
            batch["img"],
            fake,
            self.gan_type,
            gp_rng,
        )
        # d_loss is a DeviceArray, convert to float for logging
        self.current_d_loss = float(d_loss)

    def train_generator_step(self, batch_size, rng):
        z = jax.random.normal(rng, (batch_size, self.latent_dim))

        g_loss = train_generator_step_jit(
            self.generator,
            self.G_optimizer,
            self.critic,
            z,
            self.gan_type
        )
        self.current_g_loss = float(g_loss)

    def train_batch(self, batch, rng):
        for i in range(self.critic_steps):
            rng, step_rng = jax.random.split(rng)
            self.train_discriminator_step(batch, step_rng)

        rng, step_rng = jax.random.split(rng)
        self.train_generator_step(batch["img"].shape[0], step_rng)

    def save_checkpoint(self, epoch, prefix="checkpoints/"):
        os.makedirs(prefix, exist_ok=True)

        with open(f"{prefix}/generator_epoch_{epoch}.nnx", "wb") as f:
            f.write(nnx.serialization.to_bytes(self.generator))

        with open(f"{prefix}/critic_epoch_{epoch}.nnx", "wb") as f:
            f.write(nnx.serialization.to_bytes(self.critic))

        print(f"Checkpoint saved at epoch {epoch}")

    def train(self, train_loader, num_epochs, checkpoint_interval=5):
        rng = jax.random.PRNGKey(0)

        dummy_img = jnp.ones((1, 32, 32, 3))
        dummy_z = jnp.ones((1, self.latent_dim))
        _ = self.critic(dummy_img)
        _ = self.generator(dummy_z)

        self.G_optimizer = nnx.Optimizer(self.generator, self.G_tx)
        self.D_optimizer = nnx.Optimizer(self.critic, self.D_tx)

        for epoch in tqdm(range(num_epochs)):
            for batch in train_loader:
                rng, step_rng = jax.random.split(rng)
                self.train_batch(batch, step_rng)

            g_lr = (
                self.G_optimizer.step.value
                if hasattr(self.G_optimizer.step, "value")
                else 0
            )
            d_lr = (
                self.D_optimizer.step.value
                if hasattr(self.D_optimizer.step, "value")
                else 0
            )

            wandb.log(
                {
                    "D_loss": self.current_d_loss,
                    "G_loss": self.current_g_loss,
                    "epoch": epoch + 1,
                    "lr_G": g_lr,
                    "lr_D": d_lr,
                }
            )

            self.losses["critic_loss"].append(self.current_d_loss)
            self.losses["generator_loss"].append(self.current_g_loss)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"D_loss: {self.current_d_loss:.4f}, "
                f"G_loss: {self.current_g_loss:.4f}"
            )

            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1)

        return self.losses
