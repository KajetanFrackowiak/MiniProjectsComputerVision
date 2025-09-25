import wandb
import torch
import os
from tqdm import tqdm


class Trainer:
    def __init__(self, model, train_loader, optimizer, scheduler, criterion, mixup_fn,
                  alpha, psel, epochs, model_dir, device):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.mixup_fn = mixup_fn
        self.alpha = alpha
        self.psel = psel
        self.epochs = epochs
        self.model_dir = model_dir
        self.device = device

    def train_step(self, x, y, g):
        x, y_a, y_b, lam = self.mixup_fn(x, y, g, self.alpha, self.psel)
        y_a, y_b = y_a.to(self.device), y_b.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss = lam * self.criterion(outputs, y_a) + (1 - lam) * self.criterion(outputs, y_b)
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}
    
    def train_batch(self):
        batch_stats = {"total_loss": 0}

        for x, y, g in tqdm(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            g = g.to(self.device)
            loss = self.train_step(x, y, g)
            batch_stats["total_loss"] += loss.item()
        
        return batch_stats
    
    def train(self):
        self.model.train()
        train_stats = {"avg_loss": []}
        for epoch in range(self.epochs):
            batch_stats = self.train_batch()
            self.scheduler.step()

            avg_loss = batch_stats["total_loss"] / len(self.train_loader)
            train_stats["avg_loss"].append(avg_loss.item())

            print(f"Epoch: {epoch + 1}/{self.epochs}, Avg Loss: {avg_loss:.4f}")

            log_data = {
                "epoch": epoch + 1,
                **train_stats
            }
            wandb.log(log_data)
        
        os.makedirs(self.model_dir, exist_ok=True)
        torch.save({
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": epoch + 1
        }, f"{self.model_dir}/model_epoch_{epoch + 1}.pth")

        return train_stats
                
