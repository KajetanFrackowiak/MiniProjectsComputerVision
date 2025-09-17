import torch
import torch.nn.functional as F
import wandb
import os
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        train_loader,
        encoder,
        quantizer,
        decoder,
        optimizer,
        scheduler,
        epochs,
        checkpoint_interval,
        checkpoints_dir,
        reconnstruction_grid_fn,
        device,
        project_name,
    ):
        self.train_loader = train_loader
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.checkpoint_interval = checkpoint_interval
        self.checkpoints_dir = checkpoints_dir
        self.reconstructions_grid_fn = reconnstruction_grid_fn
        self.device = device
        self.project_name = project_name

        wandb.init(project=project_name)

        self.vq_losses = []
        self.recon_losses = []
        self.total_losses = []

    def train_step(self, images):
        self.encoder.train()
        self.quantizer.train()
        self.decoder.train()

        images = images.to(self.device)
        # Encoder returns (mu, logvar); take mu as the latent tensor
        z_e = self.encoder(images)[0]
        z_q, vq_loss, _ = self.quantizer(
            z_e.unsqueeze(-1).unsqueeze(-1)
        )  # add H,W for VQ, vq_loss = codebook loss + commitment loss
        x_recon = self.decoder(z_q.view(z_q.size(0), -1))


        recon_loss = F.mse_loss(x_recon, images)
        loss = recon_loss + vq_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log batch-wise losses
        wandb.log(
            {
                "vq_loss_batch": vq_loss.item(),
                "recon_loss_batch": recon_loss.item(),
                "total_loss_batch": loss.item(),
            }
        )

        return {
            "vq_loss": vq_loss.item(),
            "recon_loss": recon_loss.item(),
            "total_loss": loss.item(),
        }

    def train_epoch(self, epoch):
        epoch_metrics = {"vq_loss": 0, "recon_loss": 0, "total_loss": 0}
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.epochs}", leave=False)
        for images in pbar:
            metrics = self.train_step(images)
            for k in epoch_metrics:
                epoch_metrics[k] += metrics[k]
            pbar.set_postfix({
                "vq_loss": f"{metrics['vq_loss']:.4f}",
                "recon_loss": f"{metrics['recon_loss']:.4f}",
                "total_loss": f"{metrics['total_loss']:4f}"
            })

        n = len(self.train_loader)
        for k in epoch_metrics:
            epoch_metrics[k] /= n

        self.vq_losses.append(epoch_metrics["vq_loss"])
        self.recon_losses.append(epoch_metrics["recon_loss"])
        self.total_losses.append(epoch_metrics["total_loss"])

        return epoch_metrics

    def train(self):
        for epoch in range(self.epochs):
            epoch_metrics = self.train_epoch(epoch)
            print(f"Epoch {epoch + 1}: {epoch_metrics}")

            self.scheduler.step()

            if (epoch + 1) % self.checkpoint_interval == 0:
                os.makedirs(self.checkpoints_dir, exist_ok=True)
                checkpoint_path = os.path.join(
                    self.checkpoints_dir, f"checkpoint_epoch_{epoch + 1}.pth"
                )
                torch.save(
                    {
                        "encoder": self.encoder.state_dict(),
                        "decoder": self.decoder.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    checkpoint_path,
                )

                real_images = next(iter(self.train_loader))
                real_images = real_images[:7].to(self.device)
                with torch.no_grad():
                    z = self.encoder(real_images)[0] # mu
                    z_q, _, _ = self.quantizer(z.unsqueeze(-1).unsqueeze(-1))
                    recon_images = self.decoder(z_q.view(z_q.size(0), -1))

                self.reconstructions_grid_fn(real_images, recon_images, rows=2, cols=7, train=True, save_path=f"plots/recon/train/recon_epoch_{epoch + 1}.png")

            wandb.log(
                {
                    "epoch": epoch + 1,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    **epoch_metrics,
                }
            )

        return self.vq_losses, self.recon_losses, self.total_losses
