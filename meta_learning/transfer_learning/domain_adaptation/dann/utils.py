import os
import matplotlib.pyplot as plt

def plot(total_losses, domain_source_losses, domain_target_losses, training=True, dir_name="plots"):
    os.makedirs(dir_name, exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.plot(total_losses, color="red", linestyle="-", label="Total Loss")
    plt.plot(domain_source_losses, color="green", linestyle="--", label="Domain Source Loss")
    plt.plot(domain_target_losses, color="blue", linestyle=":", label="Domain Target Loss")
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if training:
        plt.title("DANN Training")
    else:
        plt.title("DANN Inference")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    if training:
        plt.savefig(f"{dir_name}_training.png")
    else:
        plt.savefig(f"{dir_name}_inference.png")
        