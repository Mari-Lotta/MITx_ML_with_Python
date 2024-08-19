import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from train_utils import run_epoch, batchify_data

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CNN(nn.Module):
    def __init__(self, input_dimension):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (3, 3))
        self.conv2 = nn.Conv2d(32, 64, (3, 3))
        self.conv3 = nn.Conv2d(64, 128, (3, 3))
        self.pool = nn.MaxPool2d((2, 2))
        self.flatten = Flatten()
        self.fc1 = nn.Linear(128 * 2 * 2, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2_first_digit = nn.Linear(256, 10)
        self.fc2_second_digit = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out_first_digit = self.fc2_first_digit(x)
        out_second_digit = self.fc2_second_digit(x)
        return out_first_digit, out_second_digit
def get_twodigit_data():
    # Load your two-digit MNIST dataset
    # This is a placeholder function. Replace with your actual data loading code.
    # Example assumes .npz file with 'X_train', 'y_train', 'X_test', 'y_test' arrays.
    
    with np.load('twodigit_mnist.npz') as data:
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
    
    # Normalize data and reshape to (N, 1, 28, 28)
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0
    X_train = X_train.reshape(-1, 1, 28, 28)
    X_test = X_test.reshape(-1, 1, 28, 28)
    
    return X_train, y_train, X_test, y_test

def main():
    # Load data
    X_train, y_train, X_test, y_test = get_twodigit_data()  # This function should be defined to get your data

    # Create the model
    model = CNN(input_dimension=(28, 28))

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Batchify data
    batch_size = 32
    train_batches = batchify_data(X_train, y_train, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    # Train the model
    for epoch in range(20):  # Train for 20 epochs
        run_epoch(train_batches, model.train(), criterion, optimizer)
        val_loss, val_accuracy = run_epoch(test_batches, model.eval(), criterion, None)
        print(f"Epoch {epoch + 1} - Val loss: {val_loss:.4f}, Val accuracy: {val_accuracy:.4f}")

    # Test the model
    test_loss, test_accuracy = run_epoch(test_batches, model.eval(), criterion, None)
    print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_accuracy:.4f}")

if __name__ == '__main__':
    main()
