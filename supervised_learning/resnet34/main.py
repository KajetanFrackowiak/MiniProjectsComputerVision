import argparse
import torch
from data.data_loader import get_data_loaders, load_config
from training.train import train
from inference.inference import inference
from models.resnet import ResNet34_CIFAR10

def parse_args():
    parser = argparse.ArgumentParser(description="Train or run inference on ResNet-34")
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

    model = ResNet34_CIFAR10().to(device)
    
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

        torch.save(model.state_dict(), "resnet34_trained_model.pth")
    
    elif args.mode == "inference":
        _, testloader = get_data_loaders(batch_size=config["inference"]["batch_size"])

        model.load_state_dict(torch.load("resnet34_trained_model.pth"))

        inference(model, testloader, device)
    
if __name__ == "__main__":
    main()