import numpy as np
import torch
import pickle
import cifar10


def log_softmax(x):
    """Numerically stable log-softmax: x_i - log(sum_j exp(x_j))."""
    x_max = x.max(dim=1, keepdim=True).values
    return x - x_max - torch.log(torch.exp(x - x_max).sum(dim=1, keepdim=True))


def cross_entropy_loss(log_probs, targets):
    """Negative log-likelihood: pick the log-prob of the correct class."""
    batch_size = log_probs.shape[0]
    return -log_probs[range(batch_size), targets].mean()


class SoftmaxClassifier:
    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        self.num_classes = num_classes
        self.initialize()

    def initialize(self):
        self.W = (torch.randn(self.input_size + 1, self.num_classes, dtype=torch.float32) * 0.001).requires_grad_(True)

    def predict_proba(self, X):
        """Return class probabilities (softmax of logits)."""
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
        logits = X @ self.W
        probs = torch.softmax(logits, dim=1)
        return probs

    def predict(self, X):
        """Return predicted class labels."""
        probs = self.predict_proba(X)
        return torch.argmax(probs, dim=1).numpy()

    def _prepare_data(self, X):
        """Flatten and append bias column."""
        X_flat = X.reshape(X.shape[0], -1).astype(np.float32)
        return np.hstack([X_flat, np.ones((X_flat.shape[0], 1), dtype=np.float32)])

    def fit(self, X_train, y_train, X_test=None, y_test=None, lr=0.05, epochs=32, bs=32, reg=1e-3):
        X_train_prep = self._prepare_data(X_train)
        y_train = y_train.astype(np.int64)

        X_tensor = torch.from_numpy(X_train_prep)
        y_tensor = torch.from_numpy(y_train)
        n = X_tensor.shape[0]

        if X_test is not None:
            X_test_prep = torch.from_numpy(self._prepare_data(X_test))
            y_test_np = y_test.astype(np.int64)

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            indices = torch.randperm(n)
            X_shuffled = X_tensor[indices]
            y_shuffled = y_tensor[indices]

            for ii in range((n - 1) // bs + 1):
                start_idx = ii * bs
                end_idx = start_idx + bs

                xb = X_shuffled[start_idx:end_idx]
                yb = y_shuffled[start_idx:end_idx]

                logits = xb @ self.W
                log_probs = log_softmax(logits)
                data_loss = cross_entropy_loss(log_probs, yb)

                reg_loss = reg * (self.W ** 2).sum()
                loss = data_loss + reg_loss

                loss.backward()

                with torch.no_grad():
                    self.W.data -= self.W.grad * lr
                    self.W.grad.zero_()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            metrics = {"train/loss": avg_loss, "epoch": epoch + 1}

            if X_test is not None:
                with torch.no_grad():
                    preds = torch.argmax(X_test_prep @ self.W, dim=1).numpy()
                    test_acc = (preds == y_test_np).mean()
                    metrics["test/accuracy"] = test_acc

            try:
                import wandb
                if wandb.run is not None:
                    wandb.log(metrics)
            except ImportError:
                pass

            if (epoch + 1) % 5 == 0 or epoch == 0:
                acc_str = f" | Test Acc: {metrics.get('test/accuracy', 0):.4f}" if X_test is not None else ""
                print(f"Epoch {epoch+1}/{epochs} — loss: {avg_loss:.4f}{acc_str}")

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.W.detach().numpy(), f)

    def load(self, path):
        with open(path, "rb") as f:
            w_np = pickle.load(f)
        self.W = torch.from_numpy(w_np).requires_grad_(True)


if __name__ == "__main__":
    import wandb

    cifar_root = "cifar-10-batches-py"
    X_train, y_train, X_test, y_test = cifar10.load_cifar10(cifar_root)

    learning_rates = [1e-3, 1e-2, 1e-1]
    regularizations = [1e-4, 1e-3, 1e-2]
    epochs = 32
    batch_size = 64
    input_size = 32 * 32 * 3
    num_classes = 10

    for lr in learning_rates:
        for reg in regularizations:
            name = f"lr={lr}_reg={reg}"
            print(f"\n{'='*50}")
            print(f"Training: {name}")
            print(f"{'='*50}")

            run = wandb.init(
                project="computer-vision",
                group="lab2",
                name=f"softmax-manual-{name}",
                config={"lr": lr, "reg": reg, "epochs": epochs,
                        "batch_size": batch_size, "method": "manual"},
                reinit=True,
            )

            model = SoftmaxClassifier(input_size, num_classes)
            model.fit(X_train, y_train, X_test, y_test,
                      lr=lr, epochs=epochs, bs=batch_size, reg=reg)

            X_test_prep = model._prepare_data(X_test)
            final_preds = model.predict(X_test_prep)
            final_acc = (final_preds == y_test).mean()

            wandb.log({"test/final_accuracy": final_acc})
            print(f"Final test accuracy: {final_acc:.4f}")

            wandb.finish()
