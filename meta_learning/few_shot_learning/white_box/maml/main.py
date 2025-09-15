import argparse
import torch
import torch.optim as optim

from data import load_data
from model import Model
from training import Trainer
from utils import load_hyperparameters, plot

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=5)
    
    args = parser.parse_args()
    config = load_hyperparameters()
    
    train_loader, test_loader = load_data(args.n_way, args.k_shot, args.q_query)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = Model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["outer_learning_rate"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-5)

    loss_fn = torch.nn.CrossEntropyLoss()
    trainer = Trainer(
        model, 
        optimizer,
        scheduler,
        config["inner_steps"], 
        config["inner_learning_rate"],
        train_loader,
        loss_fn,
        config["epochs"],
        device)
    
    file_name = "training_n_way_{args.n_way}_k_shot_{args.k_shot}_q_query_{args.q_query}"
    losses = trainer.train(file_name)
    plot(losses, file_name)

if __name__ == "__main__":
    main()
