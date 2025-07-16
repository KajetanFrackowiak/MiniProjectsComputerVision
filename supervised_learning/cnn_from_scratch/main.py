from lenets import LeNet5
from data import get_data
import numpy as np
import time
import matplotlib.pyplot as plt
from tqdm import tqdm


def preprocess(images):
    # Convert images to numpy arrays, normalize, reshape
    images = np.array([np.array(img) for img in images], dtype=np.float32) / 255.0
    images = images.reshape(-1, 1, 28, 28)
    return images

def main():
    print("Using CPU for training")

    images, labels = get_data()
    images = preprocess(images)
    labels = np.array(labels)

    model = LeNet5()

    epochs = 10
    batch_size = 128

    print(f"Dataset size: {len(images)}")
    print(f"Batch size: {batch_size}")
    print(f"Batches per epoch: {len(images) // batch_size}")

    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        batches = 0

        with tqdm(
            range(0, len(images), batch_size), desc=f"Epoch {epoch + 1}/{epochs}"
        ) as pbar:
            for i in pbar:
                batch_start = time.time()
                x_batch = images[i : i + batch_size]
                y_batch = labels[i : i + batch_size]
                loss = model.train_step(x_batch, y_batch)
                losses.append(loss)
                batch_time = time.time() - batch_start

                epoch_loss += loss
                batches += 1

                # Update progress bar more frequently for better feedback
                if batches % 10 == 0:
                    pbar.set_postfix(
                        {
                            "Loss": f"{loss:.4f}",
                            "Avg Loss": f"{epoch_loss / batches:.4f}",
                            "Batch Time": f"{batch_time:.2f}s",
                        }
                    )

        avg_loss = epoch_loss / batches
        print(f"Epoch {epoch + 1} completed. Average Loss: {avg_loss:.4f}")

    # Save the trained model
    model.save_model("models/lenet5_trained.pkl")
    print("Training completed and model saved!")

    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Time')
    plt.legend()
    plt.savefig("training_loss.png")
    plt.show()

if __name__ == "__main__":
    main()
