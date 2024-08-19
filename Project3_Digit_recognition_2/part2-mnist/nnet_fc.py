import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from tqdm import tqdm
sys.path.append("..")
from utils import get_MNIST_data
from train_utils import batchify_data, run_epoch, train_model

def set_seed(seed=12321):
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_data():
    num_classes = 10
    X_train, y_train, X_test, y_test = get_MNIST_data()

    dev_split_index = int(9 * len(X_train) / 10)
    X_dev = X_train[dev_split_index:]
    y_dev = y_train[dev_split_index:]
    X_train = X_train[:dev_split_index]
    y_train = y_train[:dev_split_index]

    permutation = np.array([i for i in range(len(X_train))])
    np.random.shuffle(permutation)
    X_train = [X_train[i] for i in permutation]
    y_train = [y_train[i] for i in permutation]

    return X_train, y_train, X_dev, y_dev, X_test, y_test

def run_experiment(hidden_size=128, batch_size=32, lr=0.1, momentum=0, activation=nn.ReLU()):
    set_seed()
    X_train, y_train, X_dev, y_dev, X_test, y_test = get_data()

    train_batches = batchify_data(X_train, y_train, batch_size)
    dev_batches = batchify_data(X_dev, y_dev, batch_size)
    test_batches = batchify_data(X_test, y_test, batch_size)

    model = nn.Sequential(
        nn.Linear(784, hidden_size),
        activation,
        nn.Linear(hidden_size, 10),
    )

    train_model(train_batches, dev_batches, model, lr=lr, momentum=momentum)
    _, val_accuracy = run_epoch(dev_batches, model.eval(), None)
    _, test_accuracy = run_epoch(test_batches, model.eval(), None)
    return val_accuracy, test_accuracy

def main():
    results = {}

    # Baseline
    val_acc, test_acc = run_experiment()
    results["baseline"] = (val_acc, test_acc)
    print(f"Baseline - Val accuracy: {val_acc:.4f}, Test accuracy: {test_acc:.4f}")

    # Batch size 64
    val_acc, test_acc = run_experiment(batch_size=64)
    results["batch_size_64"] = (val_acc, test_acc)
    print(f"Batch size 64 - Val accuracy: {val_acc:.4f}, Test accuracy: {test_acc:.4f}")

    # Learning rate 0.01
    val_acc, test_acc = run_experiment(lr=0.01)
    results["learning_rate_0.01"] = (val_acc, test_acc)
    print(f"Learning rate 0.01 - Val accuracy: {val_acc:.4f}, Test accuracy: {test_acc:.4f}")

    # Momentum 0.9
    val_acc, test_acc = run_experiment(momentum=0.9)
    results["momentum_0.9"] = (val_acc, test_acc)
    print(f"Momentum 0.9 - Val accuracy: {val_acc:.4f}, Test accuracy: {test_acc:.4f}")

    # LeakyReLU activation
    val_acc, test_acc = run_experiment(activation=nn.LeakyReLU())
    results["LeakyReLU_activation"] = (val_acc, test_acc)
    print(f"LeakyReLU activation - Val accuracy: {val_acc:.4f}, Test accuracy: {test_acc:.4f}")

    # Find the best performing model
    best_val_acc = max(results, key=lambda k: results[k][0])
    print(f"The best modification is: {best_val_acc} with validation accuracy {results[best_val_acc][0]:.4f}")

    # Check if the best validation accuracy model also achieved the highest test accuracy
    best_test_acc = max(results, key=lambda k: results[k][1])
    highest_test_acc_model = (best_val_acc == best_test_acc)
    print(f"Does the best validation accuracy model also achieve the highest test accuracy? {'Yes' if highest_test_acc_model else 'No'}")

if __name__ == '__main__':
    main()
