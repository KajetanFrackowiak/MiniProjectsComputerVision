import torch
import torch.optim as optim
import argparse
import wandb
from data import load_data
from gan import Discriminator, Generator
from training import Trainer
from testing import Tester
from utils import load_hyperparameters, plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--checkpoint_path", type=str)
    args = parser.parse_args()

    config = load_hyperparameters()
    
    train_loader, test_loader = load_data(config["batch_size"])
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    generator = Generator(config["latent_dim"], config["hidden_dim"], config["img_dim"])
    generator = generator.to(device)
    
    discriminator = Discriminator(config["img_dim"], config["hidden_dim"])
    discriminator = discriminator.to(device)
    
    if args.train:
        wandb.init(project=config["project_name"], name="train_run")

        gen_optimizer = optim.Adam(params=generator.parameters(), lr=config["gen_learning_rate"], betas=[config["gen_beta1"], config["gen_beta2"]])
        gen_scheduler = optim.lr_scheduler.CosineAnnealingLR(gen_optimizer, T_max=config["epochs"], eta_min=config["gen_eta_min"])

        disc_optimizer = optim.Adam(params=discriminator.parameters(), lr=config["disc_learning_rate"], betas=[config["disc_beta1"], config["disc_beta2"]])
        disc_scheduler = optim.lr_scheduler.CosineAnnealingLR(disc_optimizer, T_max=config["epochs"], eta_min=config["disc_eta_min"])

        trainer = Trainer(
            train_loader=train_loader,
            latent_dim=config["latent_dim"],
            generator=generator,
            discriminator=discriminator,
            gen_optimizer=gen_optimizer,
            disc_optimizer=disc_optimizer,
            gen_scheduler=gen_scheduler,
            disc_scheduler=disc_scheduler,
            batch_size=config["batch_size"],
            device=device,
            checkpoint_dir=config["checkpoint_dir"],
            checkpoint_interval=config["checkpoint_interval"],
            epochs=config["epochs"]
        )

        train_stats = trainer.train()
        plot(train_stats, dir_name="plots", file_name="training.png")
    else:
        wandb.init(project=config["project_name"], name="test_run")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        discriminator.load_state_dict(checkpoint["discriminator"])
        generator.load_state_dict(checkpoint["generator"])

        tester = Tester(
            test_loader=test_loader,
            generator=generator,
            discriminator=discriminator,
            latent_dim=config["latent_dim"],
            device=device,
            n_samples=config["n_samples"]
        )
        tester.test()
        

if __name__ == "__main__":
    main()