import torch
import torch.optim as optim
import argparse
from data import load_data
from vq_vae import Encoder, VectorQuantizer, Decoder
from training import Trainer
from utils import load_hyperparameters, reconstructions_grid, plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", action="store_true")
    args = parser.parse_args()

    train_loader, test_loader = load_data()

    config = load_hyperparameters()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.training:
        encoder = Encoder(config["input_dim"], config["latent_dim"]).to(device)
        quantizer = VectorQuantizer(
            config["num_embeddings"], config["embedding_dim"], config["commitment_cost"]
        ).to(device)
        decoder = Decoder(config["latent_dim"], config["output_dim"]).to(device)

        parameters = (
            list(encoder.parameters())
            + list(quantizer.parameters())
            + list(decoder.parameters())
        )
        optimizer = optim.Adam(params=parameters, lr=config["learning_rate"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=config["eta_min"]
        )

        trainer = Trainer(
            train_loader=train_loader,
            encoder=encoder,
            quantizer=quantizer,
            decoder=decoder,
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=config["epochs"],
            checkpoint_interval=config["checkpoint_interval"],
            checkpoints_dir=config["checkpoints_dir"],
            reconnstruction_grid_fn=reconstructions_grid,
            device=device,
            project_name=config["project_name"],
        )

        vq_losses, recon_losses, total_losses = trainer.train()
        plot(vq_losses, recon_losses, total_losses, train=True)


if __name__ == "__main__":
    main()
