import torch
import torch.optim as optim
import torch.nn as nn
import argparse
import yaml
from data import load_data
from flows import PlanarFlow, RadialFlow
from vae import Encoder, Decoder, VaeWithFlows, vae_flow_loss
from training import Trainer
from utils import plot


def load_hyperparameters(file_name="hyperparameters.yaml"):
    try:
        with open(file_name, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {file_name} not found.")
        exit(1)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--flow_type", required=True, choices=["planar_flow", "radial_flow"]
    )
    args = parser.parse_args()
    config = load_hyperparameters("hyperparameters.yaml")

    if args.flow_type == "planar_flow":
        flow_class = PlanarFlow
    elif args.flow_type == "radial_flow":
        flow_class = RadialFlow

    flows = nn.ModuleList(
        [flow_class(config["latent_dim"]) for _ in range(config["n_flows"])]
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.train:
        train_loader, _ = load_data(config["batch_size"])
        encoder = Encoder(config["input_dim"], config["latent_dim"])
        decoder = Decoder(config["input_dim"], config["latent_dim"])
        model = VaeWithFlows(encoder, decoder, flows).to(device)
        parameters = (
            list(encoder.parameters())
            + list(decoder.parameters())
            + list(flows.parameters())
        )
        optimizer = optim.Adam(parameters, config["learning_rate"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=config["eta_min"]
        )

        trainer = Trainer(
            train_loader,
            model,
            vae_flow_loss,
            optimizer,
            scheduler,
            config["epochs"],
            device,
            project_name=config["project_name"],
        )
        train_total_losses, train_recon_losses, train_kl_losses = trainer.train()
        plot(
            train_total_losses,
            train_recon_losses,
            train_kl_losses,
            training=True,
            flow_type=args.flow_type,
        )


if __name__ == "__main__":
    main()
