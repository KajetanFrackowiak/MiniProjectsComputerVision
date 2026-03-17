import argparse
import os

import jax
import orbax.checkpoint as ocp
from flax import nnx
from tqdm import tqdm

from load_data import celeba_preprocess, cifar10_preprocess, get_data_loader
from model import VAE
from train import eval_step
from utils import load_config, save_stats


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
    for batch in tqdm(test_loader, desc="Testing"):
        metrics, _ = eval_step(model, batch["image"], data_key)
        total_loss += float(metrics["total"])
        num_batches += 1

    avg_test_loss = total_loss / num_batches
    save_stats({"test_loss": avg_test_loss}, f"results/{dataset_name}_test_results.json")
    print(f"Test Loss: {avg_test_loss:.4f}")


if __name__ == "__main__":
    test()
