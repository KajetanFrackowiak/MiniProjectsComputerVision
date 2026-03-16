import os

import dm_pix as pix
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
from flax import nnx

import wandb
from load_data import get_data_loader
from model import VAE
from utils import load_config, save_stats


@nnx.jit
def train_step(model, optimizer, batch_images, key):
    aug_key, model_key = jax.random.split(key)
    x = pix.random_brightness(key=aug_key, image=batch_images, max_delta=0.1)
    x = pix.random_flip_left_right(aug_key, x)

    def loss_fn(model):
        recon_x, mu, logvar = model(x, model_key)
        recon_loss = jnp.mean(jnp.sum((recon_x - x) ** 2, axis=(1, 2, 3)))
        kl_loss = -0.5 * jnp.mean(jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=1))
        return recon_loss + kl_loss, {"recon": recon_loss, "kl": kl_loss}

    (loss, metrics), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
    optimizer.update(grads)
    return {"total": loss, **metrics}


@nnx.jit
def eval_step(model, batch_images, key):
    recon_x, mu, logvar = model(batch_images, key)
    recon_loss = jnp.mean(jnp.sum((recon_x - batch_images) ** 2, axis=(1, 2, 3)))
    kl_loss = -0.5 * jnp.mean(jnp.sum(1 + logvar - mu**2 - jnp.exp(logvar), axis=1))
    return {"total": recon_loss + kl_loss, "recon": recon_loss, "kl": kl_loss}, recon_x


def main():
    config = load_config("hyperparameters.yaml")
    wandb.init(project="VAE", config=config)

    master_key = jax.random.PRNGKey(0)
    model_key, data_key, train_key = jax.random.split(master_key, 3)

    model = VAE(config["latent_dim"], rngs=nnx.Rngs(model_key))
    schedule = optax.cosine_decay_schedule(
        config["learning_rate"], config["decay_steps"], alpha=config["alpha"]
    )
    tx = optax.adam(learning_rate=schedule)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    mngr = ocp.CheckpointManager(
        os.path.abspath("checkpoints"),
        ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        options=ocp.CheckpointManagerOptions(max_to_keep=5, create=True),
    )

    data_seeds = jax.random.randint(data_key, (4,), 0, 1000000)
    seed1, seed2, seed3, seed4 = [int(s) for s in data_seeds]
    train_ds = get_data_loader(config["batch_size"], seed1, seed2, split="train")
    val_ds = get_data_loader(config["batch_size"], seed3, seed4, split="test")

    train_iter = iter(train_ds)
    val_iter = iter(val_ds)

    global_step = 0
    steps_per_epoch = 500

    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(config["num_epochs"]):
        epoch_train_losses = []

        # --- TRAINING ---
        for _ in range(steps_per_epoch):
            batch = next(train_iter)
            train_key, subkey = jax.random.split(train_key)
            train_metrics = train_step(model, optimizer, batch["image"], subkey)

            loss_val = float(train_metrics["total"])
            epoch_train_losses.append(loss_val)

            if global_step % 50 == 0:
                wandb.log(
                    {f"train/{k}": float(v) for k, v in train_metrics.items()}, step=global_step
                )
            global_step += 1

        # --- VALIDATION ---
        eval_logs = []
        for i in range(10):
            val_batch = next(val_iter)
            train_key, subkey = jax.random.split(train_key)
            m, recon_samples = eval_step(model, val_batch["image"], subkey)
            eval_logs.append(m)

            if i == 0:
                vis_orig = (val_batch["image"][:8] * 255).astype(jnp.uint8)
                vis_recon = (recon_samples[:8] * 255).astype(jnp.uint8)
                combined = np.array(jnp.concatenate([vis_orig, vis_recon], axis=2))
                wandb.log(
                    {"visual/reconstructions": [wandb.Image(img) for img in combined]},
                    step=global_step,
                )

        avg_train = np.mean(epoch_train_losses)
        avg_val = {
            k: jnp.mean(jnp.array([m[k] for m in eval_logs])) for k in ["total", "recon", "kl"]
        }

        wandb.log({f"val/{k}": float(v) for k, v in avg_val.items()}, step=global_step)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(float(avg_train))
        history["val_loss"].append(float(avg_val["total"]))

        mngr.save(global_step, nnx.state(optimizer))
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val['total']:.4f}")

    os.makedirs("results", exist_ok=True)
    save_stats(history, "results/training_history.json")


if __name__ == "__main__":
    main()
