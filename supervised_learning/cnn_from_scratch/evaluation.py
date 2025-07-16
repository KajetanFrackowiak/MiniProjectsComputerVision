import numpy as np
import matplotlib.pyplot as plt
from lenets import LeNet5
from data import get_data
import random


def preprocess(images):
    """Preprocess images for model input."""
    images = np.array([np.array(img) for img in images], dtype=np.float32) / 255.0
    images = images.reshape(-1, 1, 28, 28)
    return images


def test_model(model_path="models/lenet5_trained.pkl", num_samples=16):
    """Test the trained model and visualize results."""
    # Load the data
    print("Loading test data...")
    images, labels = get_data()
    images = preprocess(images)
    labels = np.array(labels)

    # Use the last 10,000 images as test set (MNIST convention)
    test_images = images[-10000:]
    test_labels = labels[-10000:]

    # Load the trained model
    print(f"Loading model from {model_path}...")
    model = LeNet5()
    model.load_model(model_path)

    # Select random samples for visualization
    indices = random.sample(range(len(test_images)), num_samples)
    sample_images = test_images[indices]
    sample_labels = test_labels[indices]

    # Make predictions
    print("Making predictions...")
    predictions = model.predict(sample_images)
    predicted_classes = np.argmax(predictions, axis=1)

    # Calculate overall accuracy on a larger test set
    print("Calculating overall accuracy...")
    batch_size = 100
    correct = 0
    total = 0

    for i in range(
        0, min(1000, len(test_images)), batch_size
    ):  # Test on first 1000 samples
        batch_images = test_images[i : i + batch_size]
        batch_labels = test_labels[i : i + batch_size]
        batch_predictions = model.predict(batch_images)
        batch_predicted_classes = np.argmax(batch_predictions, axis=1)
        correct += np.sum(batch_predicted_classes == batch_labels)
        total += len(batch_labels)

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Visualize the results
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    fig.suptitle(f"LeNet-5 Predictions (Accuracy: {accuracy:.2%})", fontsize=16)

    for i, ax in enumerate(axes.flat):
        if i < len(sample_images):
            # Display the image
            image = sample_images[i, 0]  # Remove channel dimension for display
            ax.imshow(image, cmap="gray")

            # Add prediction and true label
            predicted = predicted_classes[i]
            true_label = sample_labels[i]
            confidence = predictions[i, predicted] * 100

            color = "green" if predicted == true_label else "red"
            ax.set_title(
                f"Pred: {predicted} ({confidence:.1f}%)\nTrue: {true_label}",
                color=color,
                fontsize=10,
            )
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    plt.savefig("model_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Show confusion matrix for the classes
    plot_confusion_matrix(
        test_labels[:1000], np.argmax(model.predict(test_images[:1000]), axis=1)
    )


def plot_confusion_matrix(true_labels, predicted_labels):
    """Plot confusion matrix."""
    from collections import defaultdict

    # Create confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for true, pred in zip(true_labels, predicted_labels):
        confusion[true][pred] += 1

    # Convert to numpy array
    matrix = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            matrix[i, j] = confusion[i][j]

    # Plot
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(10)
    plt.xticks(tick_marks, range(10))
    plt.yticks(tick_marks, range(10))

    # Add text annotations
    thresh = matrix.max() / 2.0
    for i in range(10):
        for j in range(10):
            plt.text(
                j,
                i,
                f"{int(matrix[i, j])}",
                horizontalalignment="center",
                color="white" if matrix[i, j] > thresh else "black",
            )

    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.show()


def plot_sample_predictions():
    """Plot some sample predictions with confidence scores."""
    # Load test data
    images, labels = get_data()
    images = preprocess(images)
    labels = np.array(labels)

    # Use test set
    test_images = images[-1000:]
    test_labels = labels[-1000:]

    # Load model
    model = LeNet5()
    model.load_model("models/lenet5_trained.pkl")

    # Get predictions for a few samples
    sample_indices = [0, 1, 2, 3, 4]
    sample_images = test_images[sample_indices]
    sample_labels = test_labels[sample_indices]

    predictions = model.predict(sample_images)

    # Plot predictions with confidence scores
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    for i, ax in enumerate(axes):
        # Display image
        ax.imshow(sample_images[i, 0], cmap="gray")

        # Get top 3 predictions
        pred_probs = predictions[i]
        top_indices = np.argsort(pred_probs)[-3:][::-1]

        title = f"True: {sample_labels[i]}\n"
        for j, idx in enumerate(top_indices):
            title += f"{idx}: {pred_probs[idx]:.3f}\n"

        ax.set_title(title, fontsize=8)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("detailed_predictions.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("Testing trained LeNet-5 model...")
    test_model()
    print("\nDetailed predictions:")
    plot_sample_predictions()
