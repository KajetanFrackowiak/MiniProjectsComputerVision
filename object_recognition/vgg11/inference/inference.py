import torch
import matplotlib.pyplot as plt
import numpy as np
from models.vgg11 import VGG11
from data.data_loader import get_data_loaders
from utils.logger import get_logger

# Define CIFAR-10 class names
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
    model.eval()  # Set the model to evaluation mode
    logger = get_logger()

    dataiter = iter(testloader)
    images, labels = next(dataiter)
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():  # Turn off gradients for inference
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)

    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    for i in range(9):
        ax = axes[i // 3, i % 3]
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        img = np.clip((img * 0.5) + 0.5, 0, 1)
        ax.imshow(img)

        correct = (predicted[i] == labels[i]).item()
        border_color = "green" if correct else "red"

        # Set border color around the axes
        ax.spines["top"].set_color(border_color)
        ax.spines["bottom"].set_color(border_color)
        ax.spines["left"].set_color(border_color)
        ax.spines["right"].set_color(border_color)

        # Adjust the width of the border for better visibility
        ax.spines["top"].set_linewidth(2)
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["left"].set_linewidth(2)
        ax.spines["right"].set_linewidth(2)

        # Use class_names to display the label names
        ax.set_title(
            f"Pred: {class_names[predicted[i].item()]}, True: {class_names[labels[i].item()]}"
        )
        ax.axis("on")
    plt.tight_layout()
    plt.show()

    logger.info("Inference completed successfully.")
