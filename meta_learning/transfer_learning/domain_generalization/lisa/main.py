import argparse
import torch
import torch.optim as optim
import wandb
from data import load_data, get_mixup_strategy
from lisa import LISA
from training import Trainer
from utils import load_hyperparameters, save_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--strategy", type=int, choices=[1,2,3,4,5], help=
                        "1: lisa"
                        "2: intra_label"
                        "3: intra_domain"
                        "4: standard"
                        "5: none", required=True)
    args = parser.parse_args()
    strategies = {1: "lisa", 2: "intra_label", 3: "intra_domain", 4: "standard", 5: "none"}
    strategy = strategies[args.strategy]

    config = load_hyperparameters()

    train_loader, test_loader = load_data(config["batch_size"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.train:
        wandb.init(project=config["project_name"], name="training")
        lisa = LISA()
        lisa = lisa.to(device)

        classifier_params = [
            p for name, p in lisa.named_parameters() if "fc" in name and p.requires_grad
        ]
        backbone_params = [
            p
            for name, p in lisa.named_parameters()
            if "fc" not in name and p.requires_grad
        ]

        optimizer = optim.Adam(
            [
                {"params": classifier_params, "lr": config["learning_rate"]},
                {
                    "params": backbone_params,
                    "lr": config.get(
                        "backbone_learning_rate", config["learning_rate"] * 0.1
                    ),
                },
            ]
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=config["eta_min"]
        )
        criterion = torch.nn.CrossEntropyLoss()

        mixup_strategy = get_mixup_strategy(strategy)

        trainer = Trainer(
            model=lisa,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            mixup_fn=mixup_strategy,
            alpha=config["alpha"],
            psel=config["psel"],
            epochs=config["epochs"],
            model_dir=config["model_dir"],
            device=device,
        )

        train_stats = trainer.train()
        save_stats(train_stats, stats_dir="stats/train", file_name=f"train_stats_{strategy}.json")

if __name__ == "__main__":
    main()
