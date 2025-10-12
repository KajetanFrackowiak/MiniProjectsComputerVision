import argparse
import wandb
import jax
from flax import nnx
import optax
from optax import linear_schedule
import secrets

from data import load_data
from models import Generator, Critic
from training import GANTrainer
from utils.tools import load_hyperparameters, plot, save_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--loss_fn", type=str, default="wgan", choices=["wgan", "wgan_gp"]
    )
    parser.add_argument(
        "--apply_sn",
        action="store_true",
        help="Apply spectral normalization to the critic",
    )
    args = parser.parse_args()

    print(f"Default backend: {jax.default_backend()}")

    config = load_hyperparameters("hyperparameters.yaml")
    train_loader, test_loader = load_data(config["batch_size"])

    if args.train:
        # Create RNG
        rngs = nnx.Rngs(0)

        # Initialize models
        G = Generator(
            latent_dim=config["latent_dim"],
            n_filters=config["generator_n_filters"],
            rngs=rngs,
        )

        D = Critic(
            n_filters=config["critic_n_filters"], apply_sn=args.apply_sn, rngs=rngs
        )

        # Create learning rate schedules
        G_scheduler = linear_schedule(
            init_value=config["generator_learning_rate"],
            end_value=0.0,
            transition_steps=config["generator_transition_steps"],
        )

        D_scheduler = linear_schedule(
            init_value=config["critic_learning_rate"],
            end_value=0.0,
            transition_steps=config["critic_transition_steps"],
        )

        # Initialize wandb
        wandb.init(
            project="WGANs",
            config=config,
            name=f"{args.loss_fn}_{'SN' if args.apply_sn else 'noSN'}_seed_{secrets.randbelow(2**32)}",
        )

        # Create trainer (optimizers will be created after param initialization)
        trainer = GANTrainer(
            generator=G,
            critic=D,
            generator_tx=optax.adam(G_scheduler),
            critic_tx=optax.adam(D_scheduler),
            gan_type=args.loss_fn,
            latent_dim=config["latent_dim"],
            critic_steps=config["critic_steps"],
        )

        # Train
        train_stats = trainer.train(
            train_loader(),
            num_epochs=config["num_epochs"],
            checkpoint_interval=config["checkpoint_interval"],
        )

        save_stats(train_stats, file_path="stats/training/training_stats.json")
        plot(train_stats, file_path="plots/training/training_losses.png")


if __name__ == "__main__":
    main()
