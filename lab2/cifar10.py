import os
import pickle
import numpy as np
import torch
from torch import nn, optim
import tqdm
import wandb


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


class Cifar10Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32 * 32 * 3, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.linear(x)



def rmse_loss(pred):
    """RMSE between logits and one-hot targets (not ideal for classification)."""
    target_onehot = torch.zeros_like(pred)
    target_onehot.scatter_(1, target.unsqueeze(1), 1.0)

    return torch.sqrt(torch.mean((pred - target_onehot) ** 2))


def ce_loss(pred, target):
    x_max = pred.max(dim=1, keepdim=True).values
    log_probs = pred - x_max - torch.log(torch.exp(pred - x_max).sum(dim=1, keepdim=True))

    batch_size = pred.shape[0]
    return -log_probs[range(batch_size), target].mean()

def ce_label_smoothing_loss(pred, target, epsilon=0.1):
    """Cross-entropy with label smoothing."""
    num_classes = pred.shape[1]
    x_max = pred.max(dim=1, keepdim=True).values
    log_probs = pred - x_max - torch.log(torch.exp(pred - x_max).sum(dim=1, keepdim=True))

    smooth_target = torch.full_like(pred, epsilon / (num_classes - 1))
    smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - epsilon)

    return -(smooth_target * log_probs).sum(dim=1).mean()


def hinge_loss(pred, target, margin=1.0):
    """Multiclass hinge loss: sum_j max(0, s_j - s_y + margin) for j != y."""
    correct_scores = pred[range(pred.shape[0]), target].unsqueeze(1)
    margins = torch.clamp(pred - correct_scores + margin, min=0)
    margins[range(pred.shape[0]), target] = 0
    return margins.sum(dim=1).mean()



def training(X_train, y_train, X_test, y_test, loss_func, lr, weight_decay):
    model = Cifar10Classifier()
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    epochs = 50
    bs = 64
    num_train = X_train.shape[0]

    for epoch in tqdm.tqdm(range(epochs), desc="Epochs"):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for ii in range((num_train - 1) // bs + 1):
            start_idx = ii * bs
            end_idx = start_idx + bs
            xb = X_train[start_idx:end_idx]
            yb = y_train[start_idx:end_idx]
            pred = model(xb) # call the forward function
            loss = loss_func(pred, yb) # apply the loss function

            loss.backward() # start the backpropagation and compute the gradients
            opt.step() # apply the parameter update
            opt.zero_grad() # zero out the gradients

            epoch_loss += loss.item()
            n_batches += 1

        with torch.no_grad():
            model.eval()
            train_pred = model(X_train).argmax(dim=1)
            train_acc = (train_pred == y_train).float().mean().item()
            test_pred = model(X_test).argmax(dim=1)
            test_acc = (test_pred == y_test).float().mean().item()
            test_loss = loss_func(model(X_test), y_test).item()

        if wandb.run is not None:
            wandb.log({
                "train/loss": epoch_loss / n_batches,
                "train/accuracy": train_acc,
                "test/loss": test_loss,
                "test/accuracy": test_acc,
                "epoch": epoch + 1,
            })

    print(f"  Train acc: {train_acc:.4f}  Test acc: {test_acc:.4f}")
    return train_acc, test_acc


if __name__ == "__main__":
    cifar_root_dir = "cifar-10-batches-py"
    X_train, y_train, X_test, y_test = load_cifar10(cifar_root_dir)

    # convert the training and test data to floating point
    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)

    # Reshape the training data such that we have one image per row
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))

    # pre-processing: subtract mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image

    # convert everything to tensors
    X_train, y_train, X_test, y_test = map(torch.tensor, (X_train, y_train, X_test, y_test))
    X_train = X_train.float()
    X_test = X_test.float()

    loss_functions = {
        "rmse": rmse_loss,
        "cross-entropy": ce_loss,
        "hinge": hinge_loss,
        "ce-label-smoothing": ce_label_smoothing_loss,
    }
    learning_rates = [1e-3, 1e-4, 1e-5]
    weight_decays = [0, 1e-3, 1e-4]

    for loss_name, loss_func in loss_functions.items():
        for lr in learning_rates:
            for wd in weight_decays:
                name = f"{loss_name}_lr={lr}_wd={wd}"
                print(f"\n{'='*50}")
                print(f"Training: {name}")
                print(f"{'='*50}")

                wandb.init(
                    project="computer-vision",
                    group="lab2",
                    name=f"softmax-pytorch-{name}",
                    config={"loss": loss_name, "lr": lr, "weight_decay": wd,
                            "epochs": 50, "batch_size": 64, "method": "pytorch"},
                    reinit=True,
                )

                train_acc, test_acc = training(X_train, y_train, X_test, y_test, loss_func, lr, wd)

                wandb.log({"test/final_accuracy": test_acc, "train/final_accuracy": train_acc})
                wandb.finish()
