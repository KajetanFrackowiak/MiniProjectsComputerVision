import yaml
import os
import matplotlib.pyplot as plt

def load_hyperparameters(file_name="hyperparameters.yaml"):
    try:
        with open(file_name, "r") as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {file_name} not found.")
        exit(1)
    return config

def plot(losses, title="traning"):
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(8, 8))
    
    plt.plot(losses, label="Avg Losses", linewidth=2)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    plt.savefig(f"plots/{title}.png")
    plt.close()