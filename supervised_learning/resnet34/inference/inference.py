import torch
import matplotlib.pyplot as plt
import numpy as np
from models.resnet import ResNet34_CIFAR10
from  utils.logger import get_logger

class_names = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

def inference(model, testloader, device):
    model.eval()
    logger = get_logger()
    
    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)  #dim=1 is class axis, outputs contains raw scores for each class
    
    fix, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(9):
        ax = axes[i // 3, i % 3]
        # i // 3 = 0 for i = (0, 1, 2), ROWS
        # i % 3 = (1, 2, 3) for i = (1, 2, 3) COLS
        img = images[i].cpu().numpy().transpose(1, 2, 0)  # pytorch has (C, H, W) but matplotlib exects (H, W, C) 
        img = np.clip((img * 0.5) + 0.5, 0, 1) # instead in (0, 1) is in (0.5, 1)
        ax.imshow(img)
        
        correct = (predicted[i] == labels[i]).item()
        border_color = "green" if correct else "red"

        ax.spines["top"].set_color(border_color)
        ax.spines["bottom"].set_color(border_color)
        ax.spines["left"].set_color(border_color)
        ax.spines["right"].set_color(border_color)

        ax.spines["top"].set_linewidth(2)
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["left"].set_linewidth(2)
        ax.spines["right"].set_linewidth(2)
        
        ax.set_title(
            f"Pred: {class_names[predicted[i].item()]}, True: {class_names[labels[i].item()]}"
        )
        ax.axis("on")
    plt.tight_layout()
    plt.show()
    
    logger.info("Inference completed successfully")
