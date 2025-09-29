import wandb
import torch
import os
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        optimizer,
        scheduler,
        criterion,
        mixup_fn,
        alpha,
        psel,
        epochs,
        model_dir,
        device,
    ):
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
        x = x.to(self.device)
        y_a = y_a.to(self.device)
        y_b =  y_b.to(self.device)

        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss = lam * self.criterion(outputs, y_a) + (1 - lam) * self.criterion(
            outputs, y_b
        )
        loss.backward()
        self.optimizer.step()

        return {"loss": loss.item()}

    def train_batch(self):
        batch_stats = {"total_loss": 0}

        for x, y, g in tqdm(self.train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            g = g.to(self.device)
            loss_dict = self.train_step(x, y, g)
            batch_stats["total_loss"] += loss_dict["loss"]

        return batch_stats

    def train(self):
        self.model.train()
        train_stats = {"avg_loss": []}

        unfreeze_schedule = {
            10: ["layer4"],  # Unfreeze last ResNet block at epoch 10
            20: ["layer3", "layer4"],  # Unfreeze last two blocks at epoch 20
            30: None,  # Unfreeze all at epoch 30
        }

        for epoch in range(self.epochs):
            if epoch in unfreeze_schedule:
                if unfreeze_schedule[epoch] is None:
                    print(f"Epoch {epoch}: Unfreezing entire backbone")
                    self.model.unfreeze_backbone_layers()
                else:
                    print(
                        f"Epoch {epoch}: Unfreezing layers: {unfreeze_schedule[epoch]}"
                    )
                    self.model.unfreeze_backbone_layers(unfreeze_schedule[epoch])

                self.optimizer.param_groups[0]["params"] = [
                    p for p in self.model.parameters() if p.requires_grad
                ]

            batch_stats = self.train_batch()
            self.scheduler.step()

            avg_loss = batch_stats["total_loss"] / len(self.train_loader)
            train_stats["avg_loss"].append(avg_loss)

            current_lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch: {epoch + 1}/{self.epochs}, Avg Loss: {avg_loss:.4f}, LR: {current_lr:.6f}"
            )

            log_data = {
                "epoch": epoch + 1,
                "train/avg_loss": avg_loss,
                "train/learning_rate": current_lr,
                "train/total_loss": batch_stats[
                    "total_loss"
                ],
            }
            wandb.log(log_data)

        os.makedirs(self.model_dir, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch + 1,
            },
            f"{self.model_dir}/model_epoch_{epoch + 1}.pth",
        )

        return train_stats
