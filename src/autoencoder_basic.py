"""
Autoencoder Basic - Simple Autoencoder Training Script

This script trains a basic autoencoder on transformer MLP activations,
evaluating the model using Mean Squared Error (MSE) loss.

Example usage:
    python activation_enrich/src/autoencoder_basic.py \
        --activations_path ./sample_activations.pkl \
        --hidden_dim 100 \
        --epochs 10 \
        --log_interval 1

This trains a basic autoencoder with a hidden dimension of 100,
running for 10 epochs and logging every epoch.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import pickle
import csv
import datetime
import os

class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(SimpleAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

class ActivationDataset(Dataset):
    def __init__(self, activations_path):
        with open(activations_path, 'rb') as f:
            activations = pickle.load(f)
            self.activations = torch.tensor(activations['sequence_activations'], dtype=torch.float32)
        self.mean = self.activations.mean(dim=1, keepdim=True)
        self.std = self.activations.std(dim=1, keepdim=True)
        self.activations = (self.activations - self.mean) / (self.std + 1e-7)

    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        return self.activations[idx], self.activations[idx]

def train(model, data_loader, epochs, log_interval, device):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()
    mse_records = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(data_loader)
        mse_records.append(epoch_loss)
        
        if (epoch + 1) % log_interval == 0:
            print(f"Epoch {epoch + 1}/{epochs}: MSE: {epoch_loss:.6f}")

    return mse_records

def get_args():
    parser = argparse.ArgumentParser(description="Train a basic autoencoder on transformer MLP activations.")
    parser.add_argument("--activations_path", type=str, required=True, help="Path to the .pkl file containing activations.")
    parser.add_argument("--hidden_dim", type=int, default=100, help="Number of neurons in the hidden layer.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--log_interval", type=int, default=1, help="Interval for logging metrics (in epochs).")
    return parser.parse_args()

def main():
    args = get_args()
    
    dataset = ActivationDataset(args.activations_path)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleAutoencoder(input_dim=dataset.activations.shape[1], hidden_dim=args.hidden_dim).to(device)
    
    mse_records = train(model, data_loader, epochs=args.epochs, log_interval=args.log_interval, device=device)

    # Save results
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(current_time, exist_ok=True)

    # Save training metrics
    with open(f'{current_time}/training_metrics.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'MSE'])
        for epoch, mse in enumerate(mse_records, 1):
            writer.writerow([epoch, mse])

    # Save the final model
    model_save_path = f"{current_time}/final_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    print(f"Final MSE: {mse_records[-1]:.6f}")

if __name__ == "__main__":
    main()