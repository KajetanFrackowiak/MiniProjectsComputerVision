import argparse
import os

import dm_pix as pix
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import wandb
from flax import nnx

from load_data import celeba_preprocess, cifar10_preprocess, get_data_loader
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
    parser = argparse.ArgumentParser(description="Train a VAE on CIFAR-10 or CelebA")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["uoft-cs/cifar10", "flwrlabs/celeba"],
        default="uoft-cs/cifar10",
        help="Dataset to use",
    )
    args = parser.parse_args()

    config = load_config("hyperparameters.yaml")
    wandb.init(project="VAE_" + args.dataset.split("/")[-1], config=config)

    master_key = jax.random.PRNGKey(0)
    model_key, data_key, train_key = jax.random.split(master_key, 3)

    if args.dataset == "uoft-cs/cifar10":
        model = VAE(
            latents=config["cifar10_latent_dim"],
            input_size=config["cifar10_img_size"],
            features=config["cifar10_features"],
            rngs=nnx.Rngs(model_key),
        )
        schedule = optax.cosine_decay_schedule(
            config["cifar10_learning_rate"],
            config["cifar10_decay_steps"],
            alpha=config["cifar10_alpha"],
        )
    elif args.dataset == "flwrlabs/celeba":
        model = VAE(
            latents=config["celeba_latent_dim"],
            input_size=config["celeba_img_size"],
            features=config["celeba_features"],
            rngs=nnx.Rngs(model_key),
        )
        schedule = optax.cosine_decay_schedule(
            config["celeba_learning_rate"],
            config["celeba_decay_steps"],
            alpha=config["celeba_alpha"],
        )

    tx = optax.adam(learning_rate=schedule)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    mngr = ocp.CheckpointManager(
        os.path.abspath("checkpoints"),
        ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
        options=ocp.CheckpointManagerOptions(max_to_keep=5, create=True),
    )

    if args.dataset == "uoft-cs/cifar10":
        train_split = "train[:90%]"
        val_split = "train[90%:]"
        batch_size = config["cifar10_batch_size"]
        preprocess_fn = cifar10_preprocess
        img_size = config["cifar10_img_size"]
        num_epochs = config["cifar10_num_epochs"]
    elif args.dataset == "flwrlabs/celeba":
        train_split = "train"
        val_split = "valid"
        batch_size = config["celeba_batch_size"]
        preprocess_fn = celeba_preprocess
        img_size = config["celeba_img_size"]
        num_epochs = config["celeba_num_epochs"]

    seed1, seed2 = jax.random.split(data_key, 2)
    seed1, seed2 = int(seed1[0]), int(seed2[0])
    train_ds = get_data_loader(
        dataset_path=args.dataset,
        batch_size=batch_size,
        seed=seed1,
        split=train_split,
        preprocess_fn=preprocess_fn,
        target_size=img_size,
        repeat=True,
    )

    val_ds = get_data_loader(
        dataset_path=args.dataset,
        batch_size=batch_size,
        seed=seed2,
        split=val_split,
        preprocess_fn=preprocess_fn,
        target_size=img_size,
        repeat=True,
    )

    train_iter = iter(train_ds)
    val_iter = iter(val_ds)

    global_step = 0
    if args.dataset == "uoft-cs/cifar10":
        steps_per_epoch = config["cifar10_steps_per_epoch"]
    elif args.dataset == "flwrlabs/celeba":
        steps_per_epoch = config["celeba_steps_per_epoch"]

    history = {"epoch": [], "train_loss": [], "val_loss": []}

    for epoch in range(num_epochs):
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
                vis_orig = ((val_batch["image"][:8] + 1.0) / 2.0 * 255).astype(
                    jnp.uint8
                )  # [-1, 1] -> [0, 255]
                vis_recon = ((recon_samples[:8] + 1.0) / 2.0 * 255).astype(
                    jnp.uint8
                )  # [-1, 1] -> [0, 255]
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

    dataset_name = args.dataset.split("/")[-1]
    save_stats(history, f"results/{dataset_name}_training_history.json")

    # Save the final model
    os.makedirs("models", exist_ok=True)
    final_mngr = ocp.CheckpointManager(
        os.path.abspath(f"models/{dataset_name}_model"),
        ocp.Checkpointer(ocp.PyTreeCheckpointHandler()),
    )
    final_mngr.save(0, nnx.state(model))


if __name__ == "__main__":
    main()
