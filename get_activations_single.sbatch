#!/bin/bash
#SBATCH --job-name=dms_scan       # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=1        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=32G         # memory per cpu-core (4G is default)
#SBATCH --time=00:10:00          # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:1            # number of gpus per node
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


#f=uniprotkb_human_nottn.fasta 
#f=uniprotkb_small.fasta
#/scratch/gpfs/cmcwhite/activations/uniprotkb_Human_AND_model_organism_9606_2023_08_28.fasta


#f=human_proteins_dir/ARF6_HUMAN.fasta
#f=human_proteins_dir/EGFR_HUMAN.fasta
#python $HOME/vcmsa/src/vcmsa/vcmsa_embed.py -f $f -sa -aa -m /scratch/gpfs/cmcwhite/prot_t5_xl_half_uniref50-enc/ -l -4 -3 -2 -1 -o ${f}.pkl

f=human_proteins_dir/BIG1_HUMAN.fasta
#f=human_proteins_dir/CXCL2_HUMAN.fasta
#f=human_proteins_dir/OR6C6_HUMAN.fasta
python $HOME/vcmsa/src/vcmsa/vcmsa_embed.py -f $f -sa -aa -m /scratch/gpfs/cmcwhite/prot_t5_xl_half_uniref50-enc/ -l -4 -3 -2 -1 -o ${f}.pkl

#f=human_proteins_dir/VIP_HUMAN.fasta
#python $HOME/vcmsa/src/vcmsa/vcmsa_embed.py -f $f -sa -aa -m /scratch/gpfs/cmcwhite/prot_t5_xl_half_uniref50-enc/ -l -4 -3 -2 -1 -o ${f}.pkl


#f=human_proteins_dir/SOMA_HUMAN.fasta
#python $HOME/vcmsa/src/vcmsa/vcmsa_embed.py -f $f -sa -aa -m /scratch/gpfs/cmcwhite/prot_t5_xl_half_uniref50-enc/ -l -4 -3 -2 -1 -o ${f}.pkl


#for f in uniprotkb_human_nottn_splitfasta/*;
#do
#   echo $f
   #python $HOME/vcmsa/src/vcmsa/vcmsa_embed.py -f $f -sa  -m /scratch/gpfs/cmcwhite/prot_t5_xl_half_uniref50-enc/ -l -4 -3 -2 -1 -o ${f}.maxact.pkl
#done




 
