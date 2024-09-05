# Activation enrichment



## Get activation profiles
 
The main script for collecting activation profiles is mcw_utils/hf_embed.py 


#### This is a small portion of the human proteome for testing.

f=data/uniprotkb_human_nottn.seg21.fasta

#### You'll need to download a language model locally

Each model on huggingface.co will have instructions for local download with python.
The only thing is to create a cache dir in a location outside the home directory.
The working model currently is rostlab/prot_t5_xl_half_uniref50-enc

The template in mcw_utils/save_hf_model_locally.py contains past download syntax


#### Ex. get sequence level activations (max activation across all tokens in a sequence)

python mcw_utils/hf_embed.py -f \$f -sa  -m /scratch/gpfs/cmcwhite/prot_t5_xl_half_uniref50-enc/  -o ${f}.maxact.pkl


#### Ex. get amino acid level activations 

python mcw_utils/hf_embed.py -f \$f -aa  -m /scratch/gpfs/cmcwhite/prot_t5_xl_half_uniref50-enc/  -o ${f}.aa.pkl

<i>Note that this script currently pulls all activations across all layers. It would be good to add functionality to select activations from specific layers</i>
<i>For aa embeddings, we can start with just taking the final layer activations</i>

This script outputs a .pkl file, and a .pkl.description that describes file contents, and .pkl.seqnames which lists which sequences are in the .pkl file


# Data/

These are example files

- uniprotkb_human_nottn.seg21.fasta             
- uniprotkb_human_nottn.seg21.fasta.maxact.pkl  (removed too large for github)
- uniprotkb_human_nottn.seg21.fasta.maxact.pkl.description
- uniprotkb_human_nottn.seg21.fasta.maxact.pkl.seqnames



# sbatches/

Previously used slurm scripts. Can be used as a starting template, though paths are outdated

- get_activations.sbatch
- get_activations_single.sbatch  
- sae_long.sbatch



# src/

Previously used scripts. 

- activation_enrichments.py  
- autoencoder_scaleup.py



# static/

These are static files used for enrichment analysis of neurons.

- entrez_conversion.txt is a table mapping between uniprot ids and entrez ids (for compatibility with enrichr)

- uniprot_PROSITE.txt maps between uniprot ids and PROSITE annotated protein motifs. 


## Enrichr

https://maayanlab.cloud/Enrichr/

Enrichr is a database of consistently formatted annotations for human genes
https://maayanlab.cloud/Enrichr/#libraries

Each enrichr .pkl file contains a dictionary with genesets annotated for each term.  
{'Hs ACE Inhibitor Pathway WP554 22578': ['NOS3', 'AGTR2', 'REN', 'AGT', 'KNG1'], 'Hs ACE Inhibitor Pathway WP554 30178': ['NOS3', 'AGTR2', 'REN', 'AGT', 'KNG1'], 'Hs Statin Pathway PharmGKB WP430 29996': ['CETP', 'ABCA1', 'LRP1', 'DGAT1', 'LPL', 'LCAT', 'HMGCR', 'APOA4', 'CYP7A1', 'LIPC', 'PLTP', 'APOC1', 'APOC3', 'APOA1', 'SOAT1', 'APOC2', 'APOE', 'SCARB1']}

.pkl files need to be downloaded using enrichr/scripts/download_enrichr.py


# analysis/

This was a prior analysis script for a quick look at a GO term enrichment of activations.

analyze_enrichments.R



# archive/

notes_for_activations.txt



