#!/bin/bash
#SBATCH --job-name=sae       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=16        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G         # memory per cpu-core (4G is default)
#SBATCH --time=12:00:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:4            # number of gpus per node
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=cmcwhite@princeton.edu

module purge
module load anaconda3/2020.11
module load cudatoolkit

conda activate vcmsa_env
#conda activate hf-transformers


nvidia-smi

python -m torch.utils.collect_env


#10000 -> 100000 ok
#100000 -> 500000 won't create nn.Sequential on cpu
# Will do if increase CPU allocation to 514
# The GPU out of memory, maybe use more GPU

# Start on the 10000 to 100000 problem, then work up to the larger model
# 

# The parameter to really push in the starting dim.
# Even if just double the hidden_dim
hidden_dim=20000
starting_dim=5000

# So each layer is 16384 neurons
# A good goal would be last 4 layers?
# But can start with just the last layer

#hidden_dim=10000
#starting_dim=5000

f=uniprotkb_human_nottn.seg21.fasta.maxact.pkl
python autoencoder_scaleup.py \
    --activations_path $f \
    --hidden_dim_min $hidden_dim \
    --hidden_dim_max $hidden_dim \
    --last_n_activations_min $starting_dim \
    --last_n_activations_max $starting_dim \
    --steps 100 \
    --check_interval 10 \
    --optuna_trials 500 \
    --n_jobs 4

#f=uniprotkb_human_nottn.fasta 
#f=uniprotkb_small.fasta
#/scratch/gpfs/cmcwhite/activations/uniprotkb_Human_AND_model_organism_9606_2023_08_28.fasta
