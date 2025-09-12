import torch
from tqdm import tqdm
import wandb
import os


class Trainer:
    def __init__(
        self,
        train_loader,
        model,
        loss_fn,
        optimizer,
        scheduler=None,
        epochs=10,
        device=None,
        project_name="VAE_Normalizing_Flow",
        checkpoint_dir="models",
    ):
        self.train_loader = train_loader
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.train_total_losses = []
        self.train_recon_losses = []
        self.train_kl_losses = []
        wandb.init(project=project_name)
        wandb.watch(self.model, log="parameters")

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        pixels = batch
        if isinstance(batch, dict) and "pixel_values" in batch:
            pixels = batch["pixel_values"]
        pixels = pixels.to(self.device)
        recon, mu, logvar, log_det_sum = self.model(pixels)
        loss, recon_loss, kl_loss = self.loss_fn(recon, pixels, mu, logvar, log_det_sum)
        loss.backward()
        self.optimizer.step()
        return loss.item(), recon_loss.item(), kl_loss.item(), pixels, recon

    def train_one_epoch(self, epoch):
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        pbar = tqdm(
            self.train_loader, desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False
        )
        for batch in pbar:
            loss, recon_loss, kl_loss, pixels, recon = self.train_step(batch)
            epoch_loss += loss
            epoch_recon_loss += recon_loss
            epoch_kl_loss += kl_loss
            pbar.set_postfix(
                {"total_loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss}
            )
        avg_loss = epoch_loss / len(self.train_loader)
        avg_recon_loss = epoch_recon_loss / len(self.train_loader)
        avg_kl_loss = epoch_kl_loss / len(self.train_loader)
        self.train_total_losses.append(avg_loss)
        self.train_recon_losses.append(avg_recon_loss)
        self.train_kl_losses.append(avg_kl_loss)
        if self.scheduler is not None:
            self.scheduler.step()
        wandb.log(
            {
                "epoch": epoch + 1,
                "total_loss": avg_loss,
                "recon_loss": avg_recon_loss,
                "kl_loss": avg_kl_loss,
            }
        )
        grid_size = min(8, pixels.size(0))
        imgs = torch.cat([pixels[:grid_size], recon[:grid_size]], dim=0)
        wandb.log({"reconstructions": [wandb.Image(img.cpu()) for img in imgs]})
        print(
            f"Epoch {epoch + 1}/{self.epochs} - Avg Total Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, KL Loss: {avg_kl_loss:.4f}"
        )
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(
                self.checkpoint_dir, f"model_epoch_{epoch + 1}.pth"
            )
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": epoch + 1,
                },
                checkpoint_path,
            )
        return avg_loss

    def train(self):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
        return self.train_total_losses, self.train_recon_losses, self.train_kl_losses
