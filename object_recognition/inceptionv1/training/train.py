import torch
import mlflow
import mlflow.pytorch
from tqdm import tqdm
from utils.logger import get_logger
from utils.metrics import calculate_accuracy


def train(model, trainloader, criterion, optimizer, device, num_epochs=5):
    logger = get_logger()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}", ncols=100):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)

        epoch_loss = running_loss / len(trainloader)
        epoch_acc = calculate_accuracy(predicted, labels)

        logger.info(
            f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:2f}%"
        )

        mlflow.log_metric("loss", epoch_loss, step=epoch)
        mlflow.log_metric("accuracy", epoch_acc, step=epoch)

    return model
	