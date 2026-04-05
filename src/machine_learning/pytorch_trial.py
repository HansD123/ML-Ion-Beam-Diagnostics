from typing import Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
from tqdm import tqdm


class Net(nn.Module):
    """Simple CNN that takes batches of n x n images as inputs."""

    def __init__(self, image_width: int, n_outputs: int) -> None:
        self.image_width: int = image_width
        width_after_conv = (((image_width - 2) // 2) - 2) // 2

        super().__init__()
        # Convolutional layers
        self.conv1: nn.Conv2d = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=0
        )
        self.pool1: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2: nn.Conv2d = nn.Conv2d(
            in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=0
        )
        self.pool2: nn.MaxPool2d = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # Fully connected layers
        self.flatten: nn.Flatten = nn.Flatten()
        self.fc1: nn.Linear = nn.Linear(32 * width_after_conv**2, 64)
        self.fc2: nn.Linear = nn.Linear(64, n_outputs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tensor(x)
        x = x.view(-1, 1, self.image_width, self.image_width)

        # Convolutional layers
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        # Fully-connected layers
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def ml_trial(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[torch.Tensor, "Net"]:
    """Trains on batches of 30x30 input images. Handles any sized labels."""
    MAX_EPOCHS = 1000
    BATCH_SIZE = 32

    x_train = torch.tensor(x_train, dtype=torch.float)
    y_train = torch.tensor(y_train, dtype=torch.float)
    x_test = torch.tensor(x_test, dtype=torch.float)
    y_test = torch.tensor(y_test, dtype=torch.float)
    train_data = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    model = Net(len(x_train[0]), y_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    current_epoch = 1
    best_epoch = 1
    previous_val_losses = []
    while current_epoch <= MAX_EPOCHS:
        model.train()
        for train_images, train_labels in tqdm(train_loader):
            # Reset gradients to zero
            optimizer.zero_grad()

            outputs = model(train_images)

            loss = criterion(outputs, train_labels)
            loss.backward()
            optimizer.step()

        # Computing the validation loss
        model.eval()
        with torch.no_grad():
            val_outputs = model(x_test)
            val_loss = criterion(val_outputs, y_test)

        print(f"Epoch {current_epoch}/{MAX_EPOCHS}")
        print(f"Loss: {loss.item()}")
        print(f"Val. Loss: {val_loss.item()}")
        print("")

        # Early stopping if the validation loss has not improved in the last 15 epochs
        if len(previous_val_losses) == 0 or val_loss < min(previous_val_losses):
            best_model_state = copy.deepcopy(model.state_dict())
            best_epoch = current_epoch
            previous_val_losses = [val_loss]
        else:
            previous_val_losses.append(val_loss)

        if len(previous_val_losses) > 15:
            print(f"Early Stopping: Restoring parameters from epoch {best_epoch}")
            model.load_state_dict(best_model_state)
            break

        current_epoch += 1

    # Computing the validation loss
    model.eval()
    with torch.no_grad():
        val_outputs = model(x_test)
        val_loss = criterion(val_outputs, y_test)

    return val_loss, model
