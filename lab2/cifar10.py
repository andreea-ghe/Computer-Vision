import os
import pickle
import numpy as np

LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def _unpickle(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f, encoding="bytes")


def _reshape_images(raw):
    """Reshape from (N, 3072) flat array to (N, 32, 32, 3) RGB images."""
    return raw.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)


def load_cifar10(root_dir):
    """Load all CIFAR-10 train and test data.

    Returns (X_train, y_train, X_test, y_test) where images have shape (32, 32, 3).
    """
    train_data, train_labels = [], []
    for i in range(1, 6):
        batch = _unpickle(os.path.join(root_dir, f"data_batch_{i}"))
        train_data.append(batch[b"data"])
        train_labels.append(batch[b"labels"])

    X_train = _reshape_images(np.concatenate(train_data))
    y_train = np.concatenate(train_labels)

    test_batch = _unpickle(os.path.join(root_dir, "test_batch"))
    X_test = _reshape_images(test_batch[b"data"])
    y_test = np.array(test_batch[b"labels"])

    return X_train, y_train, X_test, y_test
