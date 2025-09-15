import os
import wandb
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


def get_alpha(current_step, total_steps):
    p = current_step / total_steps
    alpha = 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0
    return alpha


class Trainer:
    def __init__(
            self, 
            model, 
            source_train_loader, 
            target_train_loader, 
            optimizer, 
            scheduler, 
            lambda_grl, 
            epochs,
            device, 
            project_name="domain_adversarial_neural_networks", 
            checkpoint_dir="checkpoints"):
        self.model = model
        self.source_train_loader = source_train_loader
        self.target_train_loader = target_train_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lambda_grl = lambda_grl
        self.epochs = epochs
        self.device = device
        self.model.to(self.device)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.total_losses = []
        self.domain_source_losses = []
        self.domain_target_losses = []
        wandb.init(project=project_name)
        wandb.watch(self.model, log="parameters")

        num_batches_per_epoch = len(source_train_loader)
        self.total_steps = epochs * num_batches_per_epoch
        self.current_step = 0
        
    def train_step(self, source_batch, target_batch, alpha):
        self.model.train()
        self.optimizer.zero_grad()

        images_source, labels_source = source_batch
        images_target, _ = target_batch

        images_source = images_source.to(self.device)
        labels_source = labels_source.to(self.device)
        images_target = images_target.to(self.device)


        batch_size_source = images_source.size(0)
        batch_size_target = images_target.size(0)

        class_output_source, domain_output_source = self.model(images_source, alpha)
        _, domain_output_target = self.model(images_target, alpha)
        class_loss = F.cross_entropy(class_output_source, labels_source)

        domain_labels_source = torch.zeros(batch_size_source).long().to(self.device)
        domain_labels_target = torch.ones(batch_size_target).long().to(self.device)

        domain_loss_source = F.cross_entropy(domain_output_source, domain_labels_source)
        domain_loss_target = F.cross_entropy(domain_output_target, domain_labels_target)

        domain_loss = (domain_loss_source + domain_loss_target) / 2
        
        total_loss = class_loss + self.lambda_grl * domain_loss
        total_loss.backward()
        self.optimizer.step()
        return total_loss.item(), domain_loss_source.item(), domain_loss_target.item()
    
    def train_one_epoch(self, epoch):
        epoch_total_loss = 0.0
        epoch_domain_loss_source = 0.0
        epoch_domain_loss_target = 0.0
        pbar = tqdm(
            zip(self.source_train_loader, self.target_train_loader), desc=f"Epoch {epoch + 1}/{self.epochs}", leave=False
        )
        for source_batch, target_batch in pbar:
            self.current_step += 1
            alpha = get_alpha(self.current_step, self.total_steps)
            total_loss, domain_loss_source, domain_loss_target = self.train_step(source_batch, target_batch, alpha)
            epoch_total_loss += total_loss
            epoch_domain_loss_source += domain_loss_source
            epoch_domain_loss_target += domain_loss_target
            pbar.set_postfix(
                {"total_loss": total_loss, "domain_loss_source": domain_loss_source, "domain_loss_target": domain_loss_target}
            )
        avg_total_loss = epoch_total_loss / len(self.source_train_loader)
        avg_domain_loss_source = epoch_domain_loss_source / len(self.source_train_loader)
        avg_domain_loss_target = epoch_domain_loss_target / len(self.source_train_loader)
        self.total_losses.append(avg_total_loss)
        self.domain_source_losses.append(avg_domain_loss_source)
        self.domain_target_losses.append(avg_domain_loss_target)
        
        self.scheduler.step()

        wandb.log(
            {
                "epoch": epoch + 1,
                "total_loss": avg_total_loss,
                "domain_source_loss": avg_domain_loss_source,
                "domain_loss_target": avg_domain_loss_target,
                "lr": self.optimizer.param_groups[0]["lr"],
            }
        )

        print(f"Epoch: {epoch+1}/{self.epochs} - Avg Total Loss: {avg_total_loss:.4f} - Avg Domain Source Loss: {avg_domain_loss_source:.4f} - Avg Domain Target Loss: {avg_domain_loss_target:.4f} - LR: {self.optimizer.param_groups[0]['lr']}")
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epoch": epoch + 1,
            }, f"{self.checkpoint_dir}/checkpoint_epoch_{epoch + 1.}.pth")
        
    
    def train(self):
        for epoch in range(self.epochs):
            self.train_one_epoch(epoch)
        
        return self.total_losses, self.domain_source_losses, self.domain_target_losses


    
        

        
