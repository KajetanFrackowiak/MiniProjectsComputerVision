import argparse
import os

import jax
import matplotlib.pyplot as plt
import numpy as np
import orbax.checkpoint as ocp
from flax import nnx
from tqdm import tqdm

from load_data import celeba_preprocess, cifar10_preprocess, get_data_loader
from model import VAE
from train import eval_step
from utils import load_config, save_stats


def save_reconstruction_grid(real_images, recon_images, save_path: str, num_images: int = 8):
    real = np.array(real_images[:num_images])
    recon = np.array(recon_images[:num_images])

    # Convert images from [-1, 1] to [0, 1] for matplotlib.
    real = np.clip((real + 1.0) / 2.0, 0.0, 1.0)
    recon = np.clip((recon + 1.0) / 2.0, 0.0, 1.0)

    fig, axes = plt.subplots(2, num_images, figsize=(2 * num_images, 4))
    for i in range(num_images):
        axes[0, i].imshow(real[i])
        axes[0, i].axis("off")
        axes[1, i].imshow(recon[i])
        axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Real", fontsize=12)
    axes[1, 0].set_ylabel("Recon", fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def test():
    parser = argparse.ArgumentParser(description="Test the VAE model on CIFAR-10 or CelebA")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["uoft-cs/cifar10", "flwrlabs/celeba"],
        default="uoft-cs/cifar10",
        help="Dataset to use",
    )
    args = parser.parse_args()

    config = load_config("hyperparameters.yaml")

    master_key = jax.random.PRNGKey(0)
    model_key, data_key = jax.random.split(master_key, 2)

    if args.dataset == "uoft-cs/cifar10":
        model = VAE(
            latents=config["cifar10_latent_dim"],
            input_size=config["cifar10_img_size"],
            features=config["cifar10_features"],
            rngs=nnx.Rngs(model_key),
        )
        test_loader = get_data_loader(
            dataset_path="uoft-cs/cifar10",
            batch_size=config["cifar10_batch_size"],
            seed=0,
            split="test",
            preprocess_fn=cifar10_preprocess,
            target_size=config["cifar10_img_size"],
            repeat=False,
        )
    elif args.dataset == "flwrlabs/celeba":
        model = VAE(
            latents=config["celeba_latent_dim"],
            input_size=config["celeba_img_size"],
            features=config["celeba_features"],
            rngs=nnx.Rngs(model_key),
        )
        test_loader = get_data_loader(
            dataset_path="flwrlabs/celeba",
            batch_size=config["celeba_batch_size"],
            seed=0,
            split="test",
            preprocess_fn=celeba_preprocess,
            target_size=config["celeba_img_size"],
            repeat=False,
        )

    dataset_name = args.dataset.split("/")[-1]
    model_path = os.path.abspath(f"models/{dataset_name}_model")
    ckpt_manager = ocp.CheckpointManager(
        model_path, ocp.Checkpointer(ocp.PyTreeCheckpointHandler())
    )

    empty_state = nnx.state(model)
    restored_state = ckpt_manager.restore(0, args=ocp.args.PyTreeRestore(item=empty_state))
    nnx.update(model, restored_state)

    total_loss = 0.0
    num_batches = 0
    grid_real = None
    grid_recon = None

    for batch in tqdm(test_loader, desc="Testing"):
        metrics, reconstructions = eval_step(model, batch["image"], data_key)

        if grid_real is None:
            grid_real = batch["image"][:8]
            grid_recon = reconstructions[:8]

        total_loss += float(metrics["total"])
        num_batches += 1

    os.makedirs("plots", exist_ok=True)

    avg_test_loss = total_loss / num_batches
    save_stats({"test_loss": avg_test_loss}, f"results/{dataset_name}_test_results.json")

    if grid_real is not None and grid_recon is not None:
        grid_path = f"plots/{dataset_name}_recon_grid_2x8.png"
        save_reconstruction_grid(grid_real, grid_recon, grid_path, num_images=8)
        print(f"Saved reconstruction grid to: {grid_path}")

    print(f"Test Loss: {avg_test_loss:.4f}")


if __name__ == "__main__":
    test()
