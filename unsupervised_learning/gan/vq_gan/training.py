import jax
import jax.numpy as jnp
from flax import nnx
import wandb
from tqdm import tqdm
import os
import pickle


class Trainer:
    """
    Trainer for VQ-GAN model
    Args:
        train_loader: function that yields batches of (images, labels)
        vq_gan: VQ-GAN model (nnx.Module)
        gen_optimizer: optimizer for generator (encoder + codebook + generator)
        disc_optimizer: optimizer for discriminator
        gen_scheduler: learning rate scheduler for generator optimizer
        disc_scheduler: learning rate scheduler for discriminator optimizer
        rng: JAX random key
        batch_size: batch size
        checkpoint_dir: directory to save checkpoints
        checkpoint_interval: save checkpoint every N epochs
        epochs: number of training epochs
        recon_weight: weight for reconstruction loss
        vq_weight: weight for vector quantization loss
        gan_weight: weight for adversarial loss
    Methods:
        train_step(real_images): perform one training step
        train_batch(): train for one epoch
        train(): train for multiple epochs
    Returns:
        train_stats: dictionary with training statistics
    """
    def __init__(
        self,
        train_loader: callable,
        vq_gan: nnx.Module,
        gen_optimizer: nnx.Optimizer,
        disc_optimizer: nnx.Optimizer,
        gen_scheduler: nnx.LRScheduler,
        disc_scheduler: nnx.LRScheduler,
        rng: jax.random.PRNGKey,
        batch_size: int,
        checkpoint_dir: str = "checkpoints",
        checkpoint_interval: int = 10,
        epochs: int = 100,
        recon_weight: float = 1.0,
        vq_weight: float = 1.0,
        gan_weight: float = 0.1,
    ):
        self.train_loader = train_loader
        self.vq_gan = vq_gan
        self.gen_optimizer = gen_optimizer
        self.disc_optimizer = disc_optimizer
        self.gen_scheduler = gen_scheduler
        self.disc_scheduler = disc_scheduler
        self.rng = rng
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.epochs = epochs
        self.recon_weight = recon_weight
        self.vq_weight = vq_weight
        self.gan_weight = gan_weight

    def reconstruction_loss(self, x_hat: jnp.ndarray, x_real: jnp.ndarray) -> jnp.ndarray:
        """L2 reconstruction loss"""
        return jnp.mean((x_hat - x_real) ** 2)

    def vq_loss(self, z_e: jnp.ndarray, z_q: jnp.ndarray) -> jnp.ndarray:
        """Vector quantization loss: commitment loss"""
        # Treat z_q as a constant on computational graph to avoid double backpropagation
        commitment_loss = jnp.mean((jax.lax.stop_gradient(z_q) - z_e) ** 2)
        return commitment_loss

    def generator_loss(self, logits_fake: jnp.ndarray) -> jnp.ndarray:
        """Generator loss: fool discriminator (binary cross-entropy)"""
        return -jnp.mean(jnp.log(jax.nn.sigmoid(logits_fake) + 1e-8))

    def discriminator_loss(self, logits_real: jnp.ndarray, logits_fake: jnp.ndarray) -> jnp.ndarray:
        """Discriminator loss: distinguish real from fake (binary cross-entropy)"""
        real_loss = -jnp.mean(jnp.log(jax.nn.sigmoid(logits_real) + 1e-8))
        fake_loss = -jnp.mean(jnp.log(1 - jax.nn.sigmoid(logits_fake) + 1e-8))
        return real_loss + fake_loss

    def train_step(self, real_images: jnp.ndarray) -> dict:
        """Train for one batch"""
        self.rng, gen_key, disc_key = jax.random.split(self.rng, 3)

        def gen_loss_fn(model) -> tuple[jnp.ndarray, tuple]:
            z_e = model.encoder(real_images)
            z_q, indices = model.codebook(z_e)

            # Gradient can still flow into z_e
            z_q_st = z_e + jax.lax.stop_gradient(z_q - z_e)

            # Gradient flows x_hat -> z_q_st -> z_e
            x_hat = model.generator(z_q_st)
            # Gradients flows gen_loss_val -> x_hat -> z_q_st -> z_e
            logits_fake = model.discriminator(x_hat) 

            # Gradient flows recon_loss -> x_hat -> z_q_st -> z_e
            recon_loss = self.reconstruction_loss(x_hat, real_images)
            # Gradient flows vq_loss_val -> z_e
            vq_loss_val = self.vq_loss(z_e, z_q)
            # Gradient flows gen_loss_val -> logits_fake
            gen_loss_val = self.generator_loss(logits_fake)

            total_loss = (
                self.recon_weight * recon_loss
                + self.vq_weight * vq_loss_val
                + self.gan_weight * gen_loss_val
            )

            return total_loss, (recon_loss, vq_loss_val, gen_loss_val, x_hat)
        
        # Compute both the value of the loss function and its gradient.
        # has_aux=True means that gen_loss_fn returns (value_to_diff, aux_data),
        # where the gradient is computed only w.r.t. value_to_diff.
        grad_fn = nnx.value_and_grad(gen_loss_fn, has_aux=True)
        (gen_total_loss, (recon_loss, vq_loss_val, gen_loss_val, x_hat)), grads = grad_fn(self.vq_gan)


        self.gen_optimizer.update(grads)

        def disc_loss_fn(model) -> jnp.ndarray:
            logits_real = model.discriminator(real_images)

            # Gradient flows logits_real -> disc_loss_val
            logits_fake = model.discriminator(jax.lax.stop_gradient(x_hat))
            
            # Gradient flows logits_fake -> disc_loss_val
            disc_loss_val = self.discriminator_loss(logits_real, logits_fake)
            return disc_loss_val

        disc_loss, disc_grads = nnx.value_and_grad(disc_loss_fn)(self.vq_gan)

        self.disc_optimizer.update(disc_grads)

        return {
            "disc_loss": float(disc_loss),
            "gen_loss": float(gen_total_loss),
            "recon_loss": float(recon_loss),
            "vq_loss": float(vq_loss_val),
            "gen_adv_loss": float(gen_loss_val),
        }

    def train_batch(self) -> dict:
        """Train for one epoch"""
        batch_stats = {
            "avg_disc_loss": 0,
            "avg_gen_loss": 0,
            "avg_recon_loss": 0,
            "avg_vq_loss": 0,
            "avg_gen_adv_loss": 0,
        }

        batch_count = 0
        for i, batch in enumerate(tqdm(self.train_loader(), desc="Training")):
            real_images, _ = batch
            losses = self.train_step(real_images)

            batch_stats["avg_disc_loss"] += losses["disc_loss"]
            batch_stats["avg_gen_loss"] += losses["gen_loss"]
            batch_stats["avg_recon_loss"] += losses["recon_loss"]
            batch_stats["avg_vq_loss"] += losses["vq_loss"]
            batch_stats["avg_gen_adv_loss"] += losses["gen_adv_loss"]

            if self.gen_scheduler is not None:
                self.gen_scheduler.step()
            if self.disc_scheduler is not None:
                self.disc_scheduler.step()

            batch_count += 1

        for key in batch_stats:
            batch_stats[key] /= batch_count

        return batch_stats

    def train(self) -> dict:
        """Train for multiple epochs"""
        train_stats = {
            "avg_disc_losses": [],
            "avg_gen_losses": [],
            "avg_recon_losses": [],
            "avg_vq_losses": [],
            "avg_gen_adv_losses": [],
        }

        for epoch in range(self.epochs):
            batch_stats = self.train_batch()

            train_stats["avg_disc_losses"].append(batch_stats["avg_disc_loss"])
            train_stats["avg_gen_losses"].append(batch_stats["avg_gen_loss"])
            train_stats["avg_recon_losses"].append(batch_stats["avg_recon_loss"])
            train_stats["avg_vq_losses"].append(batch_stats["avg_vq_loss"])
            train_stats["avg_gen_adv_losses"].append(batch_stats["avg_gen_adv_loss"])

            print(
                f"Epoch: {epoch + 1}/{self.epochs}, "
                f"Disc Loss: {batch_stats['avg_disc_loss']:.4f}, "
                f"Gen Loss: {batch_stats['avg_gen_loss']:.4f}, "
                f"Recon Loss: {batch_stats['avg_recon_loss']:.4f}, "
                f"VQ Loss: {batch_stats['avg_vq_loss']:.4f}"
            )

            log_data = {
                "epoch": epoch + 1,
                **batch_stats,
            }
            wandb.log(log_data)

            if (epoch + 1) % 10 == 0:
                self.rng, img_key = jax.random.split(self.rng)
                # Generate from random latent codes
                z = jax.random.normal(img_key, (64, 8, 8, 256))
                fake_images = self.vq_gan.generator(z)

                # Denormalize images from [-1, 1] to [0, 1]
                fake_images = fake_images * 0.5 + 0.5
                fake_images = jnp.clip(fake_images, 0, 1)

                wandb.log(
                    {"Generated Images": [wandb.Image(img) for img in fake_images[:16]]}
                )

            if (epoch + 1) % self.checkpoint_interval == 0:
                os.makedirs(self.checkpoint_dir, exist_ok=True)

                _, model_state = nnx.split(self.vq_gan)

                checkpoint = {
                    "model_state": model_state,
                    "gen_optimizer_state": self.gen_optimizer.state,
                    "disc_optimizer_state": self.disc_optimizer.state,
                    "epoch": epoch + 1,
                    "rng": self.rng,
                }

                checkpoint_path = (
                    f"{self.checkpoint_dir}/checkpoint_epoch_{epoch + 1}.pkl"
                )
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(checkpoint, f)

                print(f"Checkpoint saved to {checkpoint_path}")

        return train_stats
