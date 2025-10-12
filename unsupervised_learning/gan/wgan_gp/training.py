import jax.numpy as jnp
from flax import nnx
import jax
import wandb
import os
from tqdm import tqdm
import pickle
import orbax.checkpoint as ocp

from losses import wgan_loss, wgan_gp_loss, gradient_penalty


def clip_weights(model, clip_value=0.01):
    """Clip all weights in the model to [-clip_value, clip_value] for WGAN."""
    for name, value in vars(model).items():
        if isinstance(value, nnx.Module):
            clip_weights(value, clip_value)
        elif isinstance(value, nnx.Param):
            value.value = jnp.clip(value.value, -clip_value, clip_value)


@nnx.jit
def train_discriminator_wgan(critic, optimizer, batch_img, fake):
    def loss_fn(model):
        D_real = model(batch_img)
        D_fake = model(fake)
        d_loss, _ = wgan_loss(D_real, D_fake)
        return d_loss

    loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model)
    optimizer.update(grads)
    return loss


@nnx.jit
def train_discriminator_wgan_gp(critic, optimizer, batch_img, fake, gp_rng):
    def loss_fn(model):
        D_real = model(batch_img)
        D_fake = model(fake)
        gp = gradient_penalty(model, batch_img, fake, gp_rng)
        d_loss, _ = wgan_gp_loss(D_real, D_fake, gp)
        return d_loss

    loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model)
    optimizer.update(grads)
    return loss


@nnx.jit
def train_generator_wgan(generator, optimizer, critic, z):
    def loss_fn(model):
        fake = model(z)
        D_fake = critic(fake)
        _, g_loss = wgan_loss(jnp.zeros_like(D_fake), D_fake)
        return g_loss

    loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model)
    optimizer.update(grads)
    return loss


@nnx.jit
def train_generator_wgan_gp(generator, optimizer, critic, z):
    def loss_fn(model):
        fake = model(z)
        D_fake = critic(fake)
        _, g_loss = wgan_gp_loss(jnp.zeros_like(D_fake), D_fake, 0.0)
        return g_loss

    loss, grads = nnx.value_and_grad(loss_fn)(optimizer.model)
    optimizer.update(grads)
    return loss


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
        self.checkpointer = ocp.StandardCheckpointer()

    def train_discriminator_step(self, batch, rng):
        z_rng, gp_rng = jax.random.split(rng)
        z = jax.random.normal(z_rng, (batch["img"].shape[0], self.latent_dim))

        fake = jax.lax.stop_gradient(self.generator(z))

        if self.gan_type == "wgan":
            d_loss = train_discriminator_wgan(
                self.critic,
                self.D_optimizer,
                batch["img"],
                fake,
            )
            clip_weights(self.critic, clip_value=0.01)
        else:
            d_loss = train_discriminator_wgan_gp(
                self.critic,
                self.D_optimizer,
                batch["img"],
                fake,
                gp_rng,
            )

        self.current_d_loss = float(d_loss)

    def train_generator_step(self, batch_size, rng):
        z = jax.random.normal(rng, (batch_size, self.latent_dim))

        if self.gan_type == "wgan":
            g_loss = train_generator_wgan(
                self.generator, self.G_optimizer, self.critic, z
            )
        else:
            g_loss = train_generator_wgan_gp(
                self.generator, self.G_optimizer, self.critic, z
            )

        self.current_g_loss = float(g_loss)

    def train_batch(self, batch, rng):
        for i in range(self.critic_steps):
            rng, step_rng = jax.random.split(rng)
            self.train_discriminator_step(batch, step_rng)

        rng, step_rng = jax.random.split(rng)
        self.train_generator_step(batch["img"].shape[0], step_rng)

    def save_checkpoint(self, epoch, prefix="checkpoints/"):
        """Save models with Orbax, optimizers and epoch with pickle."""
        abs_prefix = os.path.abspath(prefix)
        os.makedirs(abs_prefix, exist_ok=True)

        checkpoint_dir = f"{abs_prefix}/checkpoint_epoch_{epoch}"

        model_state = {
            "generator": nnx.state(self.generator),
            "critic": nnx.state(self.critic),
        }
        self.checkpointer.save(checkpoint_dir, model_state)

        self.checkpointer.wait_until_finished()

        optimizer_state = {
            "generator_optimizer": nnx.state(self.G_optimizer),
            "critic_optimizer": nnx.state(self.D_optimizer),
            "epoch": epoch,
        }
        optimizer_path = f"{checkpoint_dir}.optimizers.pkl"
        with open(optimizer_path, "wb") as f:
            pickle.dump(optimizer_state, f)

        print(f"Checkpoint saved at epoch {epoch}")
        print(f"  Models (Orbax): {checkpoint_dir}")
        print(f"  Optimizers (pickle): {optimizer_path}")

    def load_checkpoint(self, checkpoint_dir):
        """Load models from Orbax, optimizers and epoch from pickle."""
        abs_checkpoint_dir = os.path.abspath(checkpoint_dir)

        model_target = {
            "generator": nnx.state(self.generator),
            "critic": nnx.state(self.critic),
        }
        restored_models = self.checkpointer.restore(
            abs_checkpoint_dir, target=model_target
        )
        nnx.update(self.generator, restored_models["generator"])
        nnx.update(self.critic, restored_models["critic"])

        optimizer_path = f"{abs_checkpoint_dir}.optimizers.pkl"
        with open(optimizer_path, "rb") as f:
            optimizer_state = pickle.load(f)

        nnx.update(self.G_optimizer, optimizer_state["generator_optimizer"])
        nnx.update(self.D_optimizer, optimizer_state["critic_optimizer"])
        epoch = optimizer_state["epoch"]

        print(f"Checkpoint loaded from epoch {epoch}")
        print(f"  Models (Orbax): {abs_checkpoint_dir}")
        print(f"  Optimizers (pickle): {optimizer_path}")
        return epoch

    def train(
        self,
        train_loader,
        num_epochs,
        checkpoint_interval=5,
        start_epoch=0,
        resume_from=None,
    ):
        rng = jax.random.PRNGKey(0)

        if self.G_optimizer is None:
            dummy_img = jnp.ones((1, 32, 32, 3))
            dummy_z = jnp.ones((1, self.latent_dim))
            _ = self.critic(dummy_img)
            _ = self.generator(dummy_z)

            self.G_optimizer = nnx.Optimizer(self.generator, self.G_tx)
            self.D_optimizer = nnx.Optimizer(self.critic, self.D_tx)

            self.generator = self.G_optimizer.model
            self.critic = self.D_optimizer.model

        if resume_from:
            start_epoch = self.load_checkpoint(resume_from)
            print(f"Resuming training from epoch {start_epoch}")

        for epoch in tqdm(
            range(start_epoch, num_epochs),
            desc="Epochs",
            initial=start_epoch,
            total=num_epochs,
        ):
            _ = self.generator(dummy_z)

            self.G_optimizer = nnx.Optimizer(self.generator, self.G_tx)
            self.D_optimizer = nnx.Optimizer(self.critic, self.D_tx)

            # Important: Update references to use the models inside the optimizers
            # This ensures self.generator and self.critic are the same objects being updated
            self.generator = self.G_optimizer.model
            self.critic = self.D_optimizer.model

        for epoch in tqdm(
            range(start_epoch, num_epochs),
            desc="Epochs",
            initial=start_epoch,
            total=num_epochs,
        ):
            epoch_d_losses = []
            epoch_g_losses = []

            batch_pbar = tqdm(train_loader(), desc=f"Epoch {epoch+1}", leave=False)
            for batch_idx, batch in enumerate(batch_pbar):
                rng, step_rng = jax.random.split(rng)
                self.train_batch(batch, step_rng)

                epoch_d_losses.append(self.current_d_loss)
                epoch_g_losses.append(self.current_g_loss)

                if batch_idx % 10 == 0:
                    batch_pbar.set_postfix(
                        {
                            "D_loss": f"{self.current_d_loss:.2f}",
                            "G_loss": f"{self.current_g_loss:.2f}",
                        }
                    )

            if len(epoch_d_losses) > 0:
                avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
                avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
            else:
                avg_d_loss = self.current_d_loss
                avg_g_loss = self.current_g_loss

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
                    "D_loss": avg_d_loss,
                    "G_loss": avg_g_loss,
                    "epoch": epoch + 1,
                    "lr_G": g_lr,
                    "lr_D": d_lr,
                }
            )

            self.losses["critic_loss"].append(avg_d_loss)
            self.losses["generator_loss"].append(avg_g_loss)

            print(
                f"Epoch {epoch + 1}/{num_epochs}, "
                f"D_loss: {avg_d_loss:.4f}, "
                f"G_loss: {avg_g_loss:.4f}"
            )

            if (epoch + 1) % checkpoint_interval == 0:
                self.save_checkpoint(epoch + 1)

        return self.losses
