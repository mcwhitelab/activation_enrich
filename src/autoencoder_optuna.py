"""
Autoencoder Slim - Basic Autoencoder Training with Optuna Optimization

This script trains a basic autoencoder on transformer MLP activations,
using Optuna for hyperparameter optimization. It evaluates the model
using Mean Squared Error (MSE) loss.

Example usage:
    python activation_enrich/src/autoencoder_slim.py \
        --activations_path ./sample_activations.pkl \
        --hidden_dim_min 50 \
        --hidden_dim_max 100 \
        --steps 100 \
        --check_interval 10 \
        --optuna_trials 5

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
import optuna

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

def train(model, data_loader, epochs, device):
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

        avg_epoch_loss = epoch_loss / len(data_loader)
        mse_records.append(avg_epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}: MSE: {avg_epoch_loss}")

    return np.mean(mse_records)

def objective(trial):
    args = get_args()
    
    hidden_dim = trial.suggest_int('hidden_dim', args.hidden_dim_min, args.hidden_dim_max)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

    dataset = ActivationDataset(args.activations_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleAutoencoder(input_dim=dataset.activations.shape[1], hidden_dim=hidden_dim).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    final_mse = train(model, data_loader, epochs=args.epochs, device=device)
    
    return final_mse

def get_args():
    parser = argparse.ArgumentParser(description="Train a basic autoencoder on transformer MLP activations.")
    parser.add_argument("--activations_path", type=str, required=True, help="Path to the .pkl file containing activations.")
    parser.add_argument("--hidden_dim_min", type=int, default=50, help="Minimum number of neurons in the hidden layer.")
    parser.add_argument("--hidden_dim_max", type=int, default=200, help="Maximum number of neurons in the hidden layer.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs.")
    parser.add_argument("--optuna_trials", type=int, default=100, help="Number of Optuna trials for hyperparameter optimization.")
    return parser.parse_args()

def main():
    args = get_args()
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.optuna_trials)

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Train the final model with the best hyperparameters
    dataset = ActivationDataset(args.activations_path)
    data_loader = DataLoader(dataset, batch_size=trial.params['batch_size'], shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleAutoencoder(input_dim=dataset.activations.shape[1], hidden_dim=trial.params['hidden_dim']).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=trial.params['learning_rate'])
    
    final_mse = train(model, data_loader, epochs=args.epochs, device=device)

    # Save results
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(current_time, exist_ok=True)

    # Save the study results
    study_results = study.trials_dataframe()
    study_results.to_csv(f'{current_time}/optuna_results.csv', index=False)

    # Save the final model
    model_save_path = f"{current_time}/final_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")
    print(f"Final MSE: {final_mse}")

if __name__ == "__main__":
    main()