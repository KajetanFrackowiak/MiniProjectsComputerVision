import torch
import torch.nn.functional as F
from tqdm import tqdm


def train(model, trainloader, optimizer, scheduler, epochs, device):
    avg_losses = []

    for epoch in tqdm(range(epochs), desc="train"):
        model.train()

        total_loss = 0
        num_samples = 0

        for batch in trainloader:
            imgs = batch["pixel_values"].to(device)
            # dataset transform guarantees images in [0,1]
            num_samples += imgs.size(0)
            optimizer.zero_grad()
            output = model(imgs)
            loss = F.binary_cross_entropy_with_logits(output, imgs, reduction="sum")
            total_loss += loss
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        avg_loss = total_loss / num_samples
        avg_losses.append(avg_loss)

        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

    torch.save(
        {"made": model.state_dict(), "optimizer": optimizer.state_dict()}, "model.pth"
    )

    return avg_losses
