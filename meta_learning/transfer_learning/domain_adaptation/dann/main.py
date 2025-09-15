import yaml
import argparse
import torch
import torch.optim as optim

from data import load_data
from dann import DANN
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
    args = parser.parse_args()

    config = load_hyperparameters()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_source_loader, test_source_loader, train_target_loader, test_target_loader = load_data(config["batch_size"])

    if args.train:
        model = DANN(input_dim=config["input_dim"], output_dim=config["output_dim"])
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=config["eta_min"])
        trainer = Trainer(
            model=model,
            source_train_loader=train_source_loader,
            target_train_loader=train_target_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            lambda_grl=config["lambda_grl"],
            epochs=config["epochs"],
            device=device,
            project_name=config["project_name"],
            checkpoint_dir=config["checkpoint_dir"]
        )
        train_total_losses, train_source_losses, train_target_losses = trainer.train()
        plot(train_total_losses, train_source_losses, train_target_losses, training=True)

if __name__ == "__main__":
    main()
    