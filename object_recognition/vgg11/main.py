import argparse
import torch
from data.data_loader import get_data_loaders, load_config
from training.train import train  # Import the training function
from inference.inference import inference
from models.vgg11 import VGG11


def parse_args():
    # Command-line argument parser for different execution modes (train/inference)
    parser = argparse.ArgumentParser(
        description="Train or run inference on VGG11 model."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "inference"],
        required=True,
        help="Mode to run the script in",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for inference"
    )
    return parser.parse_args()


def main():
    # Load configuration and arguments
    config = load_config()
    args = parse_args()

    # Setup device (CUDA if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize model
    model = VGG11().to(device)

    # If in training mode
    if args.mode == "train":
        # Load the training data for training
        trainloader, _ = get_data_loaders(batch_size=config["training"]["batch_size"])

        # Set up the criterion and optimizer here (not repeated in train.py)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        # Start training (train.py handles everything related to training)
        model = train(
            model,
            trainloader,
            criterion,
            optimizer,
            device,
            num_epochs=config["training"]["epochs"],
        )

        # Save the trained model (optional, could be handled inside train.py)
        torch.save(model.state_dict(), "vgg11_trained_model.pth")

    # If in inference mode
    elif args.mode == "inference":
        # Load the test data for inference
        _, testloader = get_data_loaders(batch_size=args.batch_size)

        # Load the trained model weights
        model.load_state_dict(torch.load("vgg11_trained_model.pth"))

        # Perform inference
        inference(model, testloader, device)


if __name__ == "__main__":
    main()
