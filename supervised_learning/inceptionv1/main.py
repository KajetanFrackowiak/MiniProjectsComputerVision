import argparse
import torch
from data.data_loader import get_data_loaders, load_config
from training.train import train
from inference.inference import inference
from models.inception import GoogLeNet


def parse_args():
    # Command-line argument parser for different excution modes (train/inference)
    parser = argparse.ArgumentParser(description="Train or run inference on GoogLeNet")
    parser.add_argument(
        "--mode",
        choices=["train", "inference"],
        required=True,
        help="Mode to run the script in",
    )

    return parser.parse_args()


def main():
    config = load_config()
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GoogLeNet().to(device)

    if args.mode == "train":
        trainloader, _ = get_data_loaders(batch_size=config["training"]["batch_size"])

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config["training"]["learning_rate"]
        )

        model = train(
            model,
            trainloader,
            criterion,
            optimizer,
            device,
            num_epochs=config["training"]["epochs"],
        )

        torch.save(model.state_dict(), "inceptionv1_trained_model.pth")

    elif args.mode == "inference":
        _, testloader = get_data_loaders(batch_size=config["inference"]["batch_size"])

        model.load_state_dict(torch.load("inceptionv1_trained_model.pth"))

        inference(model, testloader, device)


if __name__ == "__main__":
    main()
