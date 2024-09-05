import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import argparse
import pickle
import csv
from scipy.stats import kurtosis
import datetime
import optuna
import os
import optuna.visualization as vis  # Added this import
import plotly
from torch.cuda.amp import GradScaler, autocast
from time import time


def average_dot_product(matrix):
    """
    Compute the average dot product of a matrix where rows are examples and columns are neuron activations.
    with the normalization, it's just the cosine similarity 
    Args:
    - matrix (torch.Tensor): The input matrix of shape (num_examples, num_neurons).
    
    Returns:
    - avg_dot_product (float): The average dot product of the rows of the matrix.
    """
    num_examples = matrix.size(1)
    
    # Check if the number of examples is greater than 1
    if num_examples <= 1:
        print("one example?")
        #return float('nan')
    
    # Check for invalid values in the matrix
    if torch.isnan(matrix).any() or torch.isinf(matrix).any():
        print("matrix contains nans")
        #return float('nan')

    ## Normalize the matrix to prevent large values
    matrix = matrix / torch.norm(matrix, dim=1, keepdim=True)
    #.to(torch.float32) 
    # Clip the values in the matrix to a reasonable range
    #matrix = torch.clamp(matrix, min=-1e6, max=1e6)



    # Compute the dot products
    dot_products = torch.mm(matrix, matrix.t())

    # Check for invalid values in the dot products
    if torch.isnan(dot_products).any():
        print("Dot products contain NaNs.")
        return float('nan')
    if torch.isinf(dot_products).any():
        print("Dot products contain Infs.")
        return float('nan')


    # Exclude the diagonal elements (self-dot products)
    dot_products = dot_products - torch.diag(dot_products.diag())
    
    # Compute the average dot product
    avg_dot_product = torch.sum(dot_products) / (num_examples * (num_examples - 1))
    if torch.isnan(avg_dot_product):
        print("Average dot product is NaN.")
        return float('nan')
    if torch.isinf(avg_dot_product):
        print("Average dot product is Inf.")
        return float('nan')
    return avg_dot_product.item()
def xavier_uniform_init(m):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

def he_uniform_init(m):
    if isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)

def orthogonal_init(m):
    if isinstance(m, nn.Linear):
        init.orthogonal_(m.weight)
        if m.bias is not None:
            init.zeros_(m.bias)

def sparse_init(m, sparsity_level=0.999):
    if isinstance(m, nn.Linear):
        init.xavier_uniform_(m.weight)
        mask = torch.rand(m.weight.size()) > sparsity_level
        m.weight.data *= mask.float()
        if m.bias is not None:
            init.zeros_(m.bias)

class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, init_method, sparsity_level = 0.99):
        super(SparseAutoencoder, self).__init__()
        print("setup up encoder")
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.ReLU(),
        )
        print("after encoder")
        self.decoder = nn.Linear(hidden_dim, output_dim, bias=True)
        print("after decoder")
        # Can try chunked initialization
        #nn.init.orthogonal_(self.encoder[0].weight)  # Orthogonal initialization
        #nn.init.orthogonal_(self.decoder.weight)
        self._initialize_weights(init_method, sparsity_level)
        print("weights initialized")

    def _initialize_weights(self, init_method, sparsity_level):
        for m in self.modules():
            if init_method == 'xavier_uniform':
                xavier_uniform_init(m)
            elif init_method == 'he_uniform':
                he_uniform_init(m)
            elif init_method == 'orthogonal':
                orthogonal_init(m)
            elif init_method == 'sparse':
                sparse_init(m, sparsity_level)
        print(f"weights initialized with {init_method} initialization")


    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

class ActivationDataset(Dataset):
    def __init__(self, activations_path, last_n=None):
        with open(activations_path, 'rb') as f:
            activations = pickle.load(f)
            activations = activations['sequence_activations']
            print(activations.shape)
        if last_n is not None:
            self.activations = torch.tensor(activations[:, -last_n:], dtype=torch.float32)
            print(self.activations.shape)
        else:
            self.activations = torch.tensor(activations, dtype=torch.float32)
        # Compute mean and standard deviation

        self.mean = self.activations.mean(dim=1, keepdim=True)
        self.std = self.activations.std(dim=1, keepdim =True)


        epsilon = 1e-7
        # Make sure no divide by zero nans 
        # Normalize the activations
        # Check for invalid values in the matrix
        if torch.isnan(self.activations).any() or torch.isinf(self.activations).any():
            print("orig input matrix contains nans")
 
        print(f"self.activations: min={self.activations.min().item()}, max={self.activations.max().item()}, mean={self.activations.mean().item()}, std={self.activations.std().item()}")
        print(self.activations.shape)
        self.activations = (self.activations - self.mean) / (self.std + epsilon)
        print(self.activations.shape)
        if torch.isnan(self.activations).any() or torch.isinf(self.activations).any():
            print("notmr matrix contains nans")
        print(f"self.activations: min={self.activations.min().item()}, max={self.activations.max().item()}, mean={self.activations.mean().item()}, std={self.activations.std().item()}")
        if torch.isnan(self.activations).any() or torch.isinf(self.activations).any():
                        print("Data contains NaN or Inf after initial norm")



    def __len__(self):
        return len(self.activations)
    
    def __getitem__(self, idx):
        activation = self.activations[idx]
        return activation, activation

# Why isn't this on device already?
def orthogonal_regularization(model, device):
    weight = model.encoder[0].weight#.to(device)
    print(weight)
    orthogonality_penalty = torch.norm(torch.mm(weight, weight.t()) - torch.eye(weight.size(0)))
    return orthogonality_penalty


def train(model, data_loader, steps=100000, check_interval=12500, device='cpu', l1_pv=1e-6, orthogonal_pv=1e6, sparsity_pv = 100, trial=None, accumulation_steps=4):
    print("start training loop")
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.MSELoss()
    scaler = GradScaler()
    mse_records = []
    sparsity_records = []
    orthogonality_records = []
    baseline_sparsity_records = []
    autoencoder_sparsity_records = []
    baseline_orthogonality_records = []
    autoencoder_orthogonality_records = []

    trial_number = trial.number if trial is not None else 'default'
    with open(f'training_metrics_{trial_number}.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        headers = ['Step', 'Average_MSE', 'Average_Sparsity', 'Average_Orthogonality', 'Baseline_Sparsity', 'Autoencoder_Sparsity', 'Baseline_Orthogonality', 'Autoencoder_Orthogonality']
        writer.writerow(headers)

        # Create a unique directory for each run
        current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        log_dir = f'./log/{current_time}'
        os.makedirs(log_dir, exist_ok=True)

        # Set up the profiler
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(log_dir),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
        
            for step in range(steps):
                step_time = time()
                zero_time = time()
                optimizer.zero_grad()
                #print("zero time", time() - zero_time)
                dl_time = time()
                for i, (inputs, _) in enumerate(data_loader):
                    load_time = time()
                    device_time = time()
                    #print(f"Batch {i} loaded from DataLoader: min={inputs.min().item()}, max={inputs.max().item()}, mean={inputs.mean().item()}, std={inputs.std().item()}")
                    if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                        print("Data contains NaNs or Infs after loading from DataLoader")
                    
                    inputs = inputs.to(device, non_blocking=True)
                    #print(f"Batch {i} transferred to device: min={inputs.min().item()}, max={inputs.max().item()}, mean={inputs.mean().item()}, std={inputs.std().item()}")
                    if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                        print("Data contains NaNs or Infs after transfer to device")


                    #print("device time per load", time() - device_time, "load ", i)
                    if torch.isnan(inputs).any() or torch.isinf(inputs).any():
                            print("input matrix contains nans")
 
                    train_time = time()
                    with autocast():
                        # Pad inputs to match the dimension of the autoencoder activations

                        padded_inputs = torch.nn.functional.pad(inputs, (0, model.encoder[0].out_features - inputs.size(1)))
                        if torch.isnan(padded_inputs).any() or torch.isinf(padded_inputs).any():
                            print("padded inputs matrix contains nans")
                        
                        # Calculate baseline sparsity for each example

                        baseline_sparsity = torch.mean((padded_inputs > 0.01).float(), dim=1)
                        
                        outputs, encoded = model(inputs)

                        #print(f"Encoded activations range: min={encoded.min().item()}, max={encoded.max().item()}, mean={encoded.mean().item()}, std={encoded.std().item()}")
                        if torch.isnan(encoded).any() or torch.isinf(encoded).any():
                            print("Encoded activations contain NaNs or Infs")
                        

                        mse_loss = criterion(outputs, inputs) 
                        l1_raw =  torch.norm(encoded, 1)
                        l1_penalty = l1_pv * l1_raw
                        
                        # Calculate sparsity of the autoencoder activations for each example
                        autoencoder_sparsity = torch.mean((encoded > 0.01).float(), dim=1)
                        
                        # sparsity measure the number greater than 0.1
                        # So more sparse has smaller sparsity
                        # So 5 - 2, should be large positive if good. So subtract
                        sparsity_diff = (torch.mean(baseline_sparsity - autoencoder_sparsity))
                        sparsity_reward = sparsity_pv * sparsity_diff
                        #print("baseline") 
                        # Calculate orthogonality of the padded inputs
                        baseline_orthogonality = average_dot_product(padded_inputs)
                        #print("autoencoder")
                        # Calculate orthogonality of the autoencoder activations
                        autoencoder_orthogonality = average_dot_product(encoded)


                        #print(baseline_orthogonality, autoencoder_orthogonality)                           
                        # Calculate the difference to get the orthogonality reward
                        #The average orthogonality is the average cosine similariy across rows.
                        # I want this to be lower after encoding
                        # So 5 (baseline)-2(encoded), we want a large positive if good, so subtract. 

                        orthogonality_diff = (baseline_orthogonality - autoencoder_orthogonality)
                        orthogonality_reward = orthogonal_pv * orthogonality_diff

                         # Introduce heavy penalties for negative rewards
                         # Last thing we want is the encoded to be less sparse or less orthogonal
                        #sparsity_penalty = torch.where(sparsity_reward < 0, -10.0 * sparsity_reward, 0.0)
                        #orthogonality_penalty = torch.where(orthogonality_reward < 0, -10.0 * orthogonality_reward, 0.0)
                        #print(sparsity_reward)
                        # Check if the rewards are negative and apply penalties
                        if sparsity_reward < 0:
                            sparsity_penalty = -10.0 * sparsity_reward
                        else:
                            sparsity_penalty = torch.tensor(0.0, device=device)

                        if orthogonality_reward < 0:
                            orthogonality_penalty = -10.0 * orthogonality_reward
                        else:
                            orthogonality_penalty = torch.tensor(0.0, device=device)
            

                        # Adjust the loss to include the sparsity and orthogonality rewards and penalties
                        #loss = mse_loss + l1_penalty - sparsity_reward - orthogonality_reward + sparsity_penalty + orthogonality_penalty
                        loss = mse_loss - 100*sparsity_reward #- 100*orthogonality_reward + sparsity_penalty + orthogonality_penalty
  
                        #print("mse_loss", mse_loss, "l1 raw", l1_raw, "sparsity_diff", sparsity_diff, "orthogonality_diff", orthogonality_diff)#, "sparsity_penalty", sparsity_penalty, "orthogonality_penalty", orthogonality_penalty)
                        # Adjust the loss to inclpude the sparsity and orthogonality rewards
                        
                        # Adjust the loss to include the sparsity and orthogonality rewards
                        #alpha = 1
                        #beta = 1e-6
                        #gamma = 1e6
                        #delta = 100

                        #loss = mse_loss + l1_penalty - sparsity_reward - orthogonality_reward

                        #loss = alpha * mse_loss + beta * l1_penalty + gamma * orthogonality_reward - delta * sparsity_reward
                        #print("loss", loss)
                        #print("mse_loss", alpha *mse_loss, "l1 penalty", beta*l1_penalty, "sparsity", delta*sparsity_reward, "orthognality", gamma *orthogonality_reward)
                                  
                        #loss = mse_loss + l1_penalty - sparsity_reward + orthogonality_reward
                        
                    scaler.scale(loss / accumulation_steps).backward()
                    if (i + 1) % accumulation_steps == 0:
                        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                if (step + 1) % check_interval == 0:
                    record_time = time()
                    mse_records.append(mse_loss.item())
                    sparsity_measure = torch.mean((encoded > 0.01).float()).item()
                    orthogonality_measure = average_dot_product(encoded)
                    sparsity_records.append(sparsity_measure)
                    orthogonality_records.append(orthogonality_measure)
                    baseline_sparsity_records.append(baseline_sparsity.mean().item())
                    autoencoder_sparsity_records.append(autoencoder_sparsity.mean().item())
                    baseline_orthogonality_records.append(baseline_orthogonality)
                    autoencoder_orthogonality_records.append(orthogonality_measure)
 
                    avg_mse = np.mean(mse_records[-check_interval:])
                    avg_sparsity = np.mean(sparsity_records[-check_interval:])
                    avg_orthogonality = np.mean(orthogonality_records[-check_interval:])
                    avg_baseline_sparsity = np.mean(baseline_sparsity_records[-check_interval:])
                    avg_autoencoder_sparsity = np.mean(autoencoder_sparsity_records[-check_interval:])
                    avg_baseline_orthogonality = np.mean(baseline_orthogonality_records[-check_interval:])
                    avg_autoencoder_orthogonality = np.mean(autoencoder_orthogonality_records[-check_interval:])
                    
                    writer.writerow([step + 1, avg_mse, avg_sparsity, avg_orthogonality, avg_baseline_sparsity, avg_autoencoder_sparsity, avg_baseline_orthogonality, avg_autoencoder_orthogonality])
                    print(f"Step {step + 1}: Average MSE: {avg_mse}, Average Sparsity: {avg_sparsity}, Average Orthogonality: {avg_orthogonality}, Baseline Sparsity: {avg_baseline_sparsity}, Autoencoder Sparsity: {avg_autoencoder_sparsity}, Baseline Orthogonality: {avg_baseline_orthogonality}, Autoencoder Orthogonality: {avg_autoencoder_orthogonality}, Step time: {step_time}")
                prof.step()  # Step the profiler
    return mse_records, sparsity_records, orthogonality_records, baseline_sparsity_records, autoencoder_sparsity_records, baseline_orthogonality_records, autoencoder_orthogonality_records            
    




# Want to do this on GPU not using np
def identify_dead_neurons(activation_records, threshold=0.001):
    # Ensure the activation records are on the GPU
    activation_records = activation_records.to('cuda')
    
    # Calculate variances along the specified axis
    variances = torch.var(activation_records, dim=0)
    
    # Identify dead neurons based on the threshold
    dead_neurons = torch.where(variances < threshold)[0]
    
    print(f"Activation records shape: {activation_records.shape}")
    print(f"Variances shape: {variances.shape}")
    print(f"Number of dead neurons: {dead_neurons.shape[0]}")
    
    return dead_neurons

# This takes some time
def handle_dead_neurons(model, dead_neurons):
    with torch.no_grad():
        for idx in dead_neurons:
            model.encoder[0].weight.data[idx].fill_(0.01)
            model.encoder[0].bias.data[idx].fill_(0.0)
            #print(f"Reset neuron {idx}")

def get_args():
    parser = argparse.ArgumentParser(description="Train a sparse autoencoder on transformer MLP activations.")
    parser.add_argument("--activations_path", type=str, required=True, help="Path to the .pkl file containing activations.")
    parser.add_argument("--hidden_dim_min", type=int, default=50, help="Minimum number of neurons in the hidden layer of the autoencoder.")
    parser.add_argument("--hidden_dim_max", type=int, default=200, help="Maximum number of neurons in the hidden layer of the autoencoder.")
    parser.add_argument("--l1_pv_min", type=float, default=1e-7, help="Minimum L1 penalty amount for sparsity regularization.")
    parser.add_argument("--l1_pv_max", type=float, default=1e-4, help="Maximum L1 penalty amount for sparsity regularization.")
    parser.add_argument("--last_n_activations_min", type=int, default=10, help="Minimum number of last activations to select from each example.")
    parser.add_argument("--last_n_activations_max", type=int, default=50, help="Maximum number of last activations to select from each example.")
    parser.add_argument("--orthogonal_pv_min", type=float, default=1, help="Minimum orthogonal penalty value.")
    parser.add_argument("--orthogonal_pv_max", type=float, default=1e6, help="Maximum orthogonal penalty value.")
    parser.add_argument("--sparsity_pv_min", type=float, default=1, help="Minimum sparsity penalty value.")
    parser.add_argument("--sparsity_pv_max", type=float, default=1e4, help="Maximum sparsity penalty value.")
    #parser.add_argument("--orthogonal_penalty_min", type=float, default=0.001, help="Minimum orthogonal penalty value.")
    #parser.add_argument("--orthogonal_penalty_max", type=float, default=0.1, help="Maximum orthogonal penalty value.")
    parser.add_argument("--steps", type=int, default=100000, help="Number of training steps.")
    parser.add_argument("--check_interval", type=int, default=12500, help="Interval for checking and logging metrics.")
    parser.add_argument("--optuna_trials", type=int, default=100, help="Number of Optuna trials for hyperparameter optimization.")  
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of parallel jobs for Optuna.")

    args = parser.parse_args()
    return args


def objective(trial):
    args = get_args()
    print("STEPS", args.steps)
    print(args.activations_path)
    print("last_n_activations", args.last_n_activations_max)
    print("hidden_dim", args.hidden_dim_max)

    hidden_dim = args.hidden_dim_min
    last_n_activations = args.last_n_activations_min
    l1_pv = 0
    print("check", args.check_interval)
    #hidden_dim = trial.suggest_int('hidden_dim', args.hidden_dim_min, args.hidden_dim_max)
    #l1_pv = trial.suggest_float('l1_pv', args.l1_pv_min, args.l1_pv_max)
    #last_n_activations = trial.suggest_int('last_n_activations', args.last_n_activations_min, args.last_n_activations_max)
    orthogonal_pv = trial.suggest_float('orthogonal_pv', args.orthogonal_pv_min, args.orthogonal_pv_max)
    sparsity_pv = trial.suggest_float('sparsity_pv', args.sparsity_pv_min, args.sparsity_pv_max)
    #init_method = trial.suggest_categorical('init_method', ['xavier_uniform', 'he_uniform', 'orthogonal', 'sparse'])
    init_method = trial.suggest_categorical('init_method', ['xavier_uniform', 'orthogonal', 'sparse'])
    #init_method = "sparse"

    dataset = ActivationDataset(args.activations_path, last_n=last_n_activations)
    print("dataset made")
    data_loader = DataLoader(
        dataset, 
        #batch_size=256, 
        batch_size=32,
        shuffle=True, 
        num_workers=0#,  # Because no complex preprocessing, disable workers make 10x faster epoch
        #pin_memory=True#,  # Pin memory
        #prefetch_factor=8  # Prefetch data
    )
    sparsity_level = 1 - (20/( hidden_dim ))
    print("sparsity level", sparsity_level)
    print("data_loader made")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = SparseAutoencoder(input_dim=len(dataset.activations[0]), hidden_dim=hidden_dim, output_dim=len(dataset.activations[0]), init_method = init_method, sparsity_level = sparsity_level  )
    print("model on CPU")
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    print("model parallelized")
    print(f"Let's use {torch.cuda.device_count()} GPUs!")
    model = model.to(device)
    print("model initialized")
    mse_records, sparsity_records, orthogonality_records, baseline_sparsity_records, autoencoder_sparsity_records, baseline_orthogonality_records, autoencoder_orthogonality_records = train(
    model, data_loader, steps=args.steps, check_interval=args.check_interval, device=device, 
    l1_pv=l1_pv, orthogonal_pv=orthogonal_pv, sparsity_pv=sparsity_pv, trial=trial
    )
    
    # Define a composite objective to minimize (e.g., weighted sum of MSE and sparsity)

    # Ok, I want to still do sparsity difference

    final_mse = np.mean(mse_records[-args.check_interval:])
    final_sparsity = np.mean(sparsity_records[-args.check_interval:])
    #composite_objective = 0.5 * final_mse + 1 * final_sparsity  # Adjust weights based on importance
    composite_objective = final_sparsity

    return composite_objective

def main():
    args = get_args()
    study = optuna.create_study(direction='minimize')
    print("study created")
    study.optimize(objective, n_trials=args.optuna_trials, n_jobs = args.n_jobs)  # Use the new argument
    # Print the best trial
    print("Best trial:")
    trial = study.best_trial

    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Convert study trials to a DataFrame
    df = study.trials_dataframe()

    # Create a folder named by the current datetime
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    os.makedirs(current_time, exist_ok=True)

    # Save the DataFrame to a CSV file
    df.to_csv(f"{current_time}/trials_dataframe.csv", index=False)

    # Generate and save visualizations
    # Plot optimization history
    fig = vis.plot_optimization_history(study)
    fig.write_image(f"{current_time}/optimization_history.png")

    # Plot parallel coordinate
    fig = vis.plot_parallel_coordinate(study)
    fig.write_image(f"{current_time}/parallel_coordinate.png")

    # Plot parameter importances
    fig = vis.plot_param_importances(study)
    fig.write_image(f"{current_time}/param_importances.png")

    # Plot slice
    fig = vis.plot_slice(study)
    fig.write_image(f"{current_time}/slice.png")

    # Plot contour
    fig = vis.plot_contour(study)
    fig.write_image(f"{current_time}/contour.png")

    # Train the final model with the best hyperparameters
    dataset = ActivationDataset(args.activations_path, last_n=last_n_activations)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8)  # Use multiple workers
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SparseAutoencoder(input_dim=len(dataset.activations[0]), hidden_dim=trial.params['hidden_dim'], output_dim=len(dataset.activations[0])).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    model.to(device)
    
    train(
        model, data_loader, steps=args.steps, check_interval=args.check_interval, device=device, 
        l1_pv=trial.params['l1_pv'], orthogonal_pv=trial.params['orthogonal_pv'], sparsity_pv=trial.params['sparsity_pv']
    )    

    # Save the final model
    model_save_path = f"{current_time}/final_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()