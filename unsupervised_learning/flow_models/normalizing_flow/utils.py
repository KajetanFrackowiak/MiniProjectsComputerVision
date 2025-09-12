import os
import matplotlib.pyplot as plt

def plot(total_losses, recon_losses, kl_losses, training=True, flow_type="Planar"):
    os.makedirs("plots", exist_ok=True) 

    plt.figure(figsize=(8, 8))
    plt.plot(total_losses, label="Total Loss", color="red")
    plt.plot(recon_losses, label="Recon Loss", color="green")
    plt.plot(kl_losses, label="KL Losses", color="blue")
    if training:
        plt.title(f"Training Losses, Flow Type: {flow_type}")
    else:
        plt.title(f"Inference Time, Flow Type: {flow_type}")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    file_subname = "training" if training else "inference"
    plt.savefig(f"plots/{file_subname}_{flow_type}.png")
    plt.close()