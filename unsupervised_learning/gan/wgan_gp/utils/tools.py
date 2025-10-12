import yaml
import os
import matplotlib.pyplot as plt
import json

def load_hyperparameters(config_path: str):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def save_stats(stats, file_path='training_stats.json'):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(stats, f)

def load_stats(file_path='training_stats.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            stats = json.load(f)
        return stats
    else:
        return None

def plot(losses, file_path='training_losses.png'):
    plt.figure(figsize=(10, 5))
    plt.plot(losses['critic_loss'], label='Critic Loss')
    plt.plot(losses['generator_loss'], label='Generator Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Losses')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()