library(reticulate)
library(plotly)
# Remember to request 128GB at ele

#reticulate::install_python()

#py_install("numpy")

activations <- reticulate::py_load_object("/scratch/gpfs/cmcwhite/activations/uniprotkb_small.fasta.pkl")


head(activations$sequence_activations)


activations$sequence_activations[,2]

library(readr)
library(dplyr)
library(tidyr)
library(ggplot2)
conversion <- read_tsv("/scratch/gpfs/cmcwhite/activations/entrez_conversion.txt", col_names = c("Entry", "protname")) %>% filter(!is.na(protname))  %>% filter(!is.na(Entry))

prot_names <- read_csv("/scratch/gpfs/cmcwhite/activations/uniprotkb_small.fasta.pkl.seqnames", col_names = "uniprot_id")
#prot_names <- prot_names %>% rowid_to_column()
prot_names <- prot_names %>% separate(uniprot_id, into = c("tmp", "Entry", "Entry_name"), sep = "[|]") %>% select(-tmp)# %>%#inner_join(conversion)


# Checking the distribution of activations






test <- activations$sequence_activations[,1:4]


df <- conversion %>% inner_join(bind_cols(prot_names, as_tibble( activations$sequence_activations)))

one_neuron <- df %>% select(protname, Entry_name, paste0("V", 389064 + 1))


one_neuron_enrichments <- read_csv("/scratch/gpfs/cmcwhite/activations/output_uniprotkb_small.fasta/uniprotkb_small.fasta.neuron389064.libGO_Molecular_Function_2023.thresh0.01_ligands")
one_neuron_enrichments_long <- one_neuron_enrichments %>% 
  filter(nes > 0) %>% 
  separate_rows(leading_edge, sep = ",") %>%
  rename(protname = leading_edge) %>%
  left_join(conversion)

one_neuron %>%
  full_join(one_neuron_enrichments_long ) %>% #View()
  ggplot(aes(x = V389065, group = Term, color = Term)) + 
  
  geom_density()

ggplotly()


# One protein activations
soma_activations <- reticulate::py_load_object("/scratch/gpfs/cmcwhite/activations/human_proteins_dir/SOMA_HUMAN.fasta.pkl")


x <- as_tibble(soma_activations$aa_activations)


#cytokine_activity <- str_split("CXCL2,IL10,CCL16,IL4,LIF,IFNB1,IFNL4,IL12A,IL15,CCL22,XCL1,CXCL12,IFNA7,CCL25,IFNG,CXCL8,CSF3,IL5,CCL3,IFNA17,CCL2,CCL24,IL6,IFNA4,CCL4,IFNA10,IFNA5,IFNA2,IFNA21,IFNA8,CCL11,IFNA1,IL3,IFNA14,IL11,EDN1,CCL3L1,KITLG,CCL5,IFNA6,CNTF,PPBP,EPO,CXCL11,CCL1,OSM,IFNW1,CXCL13,IL9,PF4V1,CXCL10,CXCL5,IL7,CSF1,CCL21,CXCL1,CXCL3,CSF2,PF4", pattern = ",")
#receptor_ligand_activity <- str_split(, pattern = ",")
#chemokine_receptor_binding <- str_split(, pattern = ",")
#neuropeptide_hormone_activity <- str_split(, pattern = ",")
#hormone_activity <- str_split(, pattern = ",")
#growth_factor_activity <- str_split(, pattern = ",")

hormone_list <-  c("VIP", "EDN2", "SCT", "CGA", "TSHB", "FSHB", "GNRH1", "CCL25", "PTHLH", "EDN3", "PYY", "APELA", "NPPB", "VGF", "GAST", "EDN1", "PPY", "CRH", "IGF1", "INS", "LEP", "PTH", "CHGB", "CCK")
receptors <- c("ABCA1", "CLCN5", "KIF2A", "MSH6", "HSP90AA1", "CLCN3", "MYH10", "CLCN7", "MYH9", "ABCD1", "KIF21B", "SMARCA5", "MYH8", "MYH6", "MYH7", "SLC26A4", "CARNS1", "MSH2", "ABCD3", "HSPD1", "DDX3X", "KIF1C", "KIF3C", "KIF1B", "HSPA5", "CLCA1", "NSF", "DHX16", "SLC26A3", "DNA2", "HSPA1A", "ERCC3", "CFTR", "ATP1A2", "ABCC1", "HSPA1B", "HSPA8", "CLCN4")
# What is musashi doing on this list?


all_hormones <-  c("AGRP", "NPFF", "VGF", "CCL25", "HCRT", "STC2", "BMP10", "UTS2", "AGT", "KNG1", "GNRH1", "NPPA", "OXT", "AVP", "POMC", "PENK", "CGA", "TSHB", "FSHB", "LHB", "PRL", "GH1", "GH2", "CALCA", "TG", "PTH", "GCG", "VIP", "GHRH", "PPY", "NPY", "INS", "IGF2", "GAST", "EPO", "TTR", "AMH", "RLN2", "IGF1", "CHGB", "INHA", "EDN1", "CCK", "CRH", "CALCA", "GRP", "INHBA", "NMB", "INHBB", "GIP", "SCT", "APELA", "CSH1", "CSH2", "CGB3", "CGB5", "CGB8", "PYY", "CALCB", "IAPP", "PTHLH", "EDN3", "NPPB", "ADCYAP1", "TRH", "EDN2", "GAL", "NPPC", "NTS", "ADM", "FBN1", "FBN2", "THPO", "LEP", "ASIP", "INSL3", "STC1", "COPA", "UCN", "SST", "HAMP", "PRLH", "GUCA2A", "REG3A", "PNOC", "CSHL1", "ADIPOQ", "CARTPT", "C1QTNF12", "ANGPTL8", "ADM2", "GPHB5", "TOR2A", "FNDC5", "RLN3", "UCN3", "UCN2", "GPHA2", "SPX", "ECRG4", "RETN", "GALP", "GHRL", "KL", "APLN", "INSL5", "CORT", "GNRH2", "RLN1", "C1QTNF9", "CGB7", "MLN", "PMCH", "INHBC", "INHBE", "OSTN", "QRFP", "INSL4", "ERFE", "METRNL", "RETNLB", "CGB1", "PMCHL1", "CGB2", "ENHO", "UTS2B", "PMCHL2", "METRN", "INSL6", "INS-IGF2", "PYY3")

# Ok, not really receptors, things involve in microtubule motors? Go annotations seem wrong, wait for 2023 set

annot <- one_neuron %>% mutate(set = case_when(#protname %in% hormone_list ~ "hormone", 
                                      protname %in% receptors ~ "receptors",
                                      protname %in% all_hormones ~ "hormones",
                                      TRUE ~ "other"))

annot %>%
  ggplot(aes(x = V389065, group = set, color = set)) + 
  
     geom_density()

annot %>% filter(set == "hormones") %>% View()

annot %>% #arrange(desc(V389065 )) %>%
  group_by(set) %>%
     summarize(med = median(V389065), sd = sd(V389065)) %>%
  ungroup()


# Almost everything in the range filter(V389065 > -5000) %>% filter(V389065 < -800) has a GO annotation for protein binding.
# This range of activation of this neuron are all hormones/cytokines/receptor binders/protein inhibitors. 

# The thing is, this is kind of hard to disentangle. Does the near 0 count as a high amount of activation or low. 
# Ok, for these activations I took the sum. hmm. Will look at an individual aa level later
# Also recalculate with max, and also min. Add as a command line options.

# Interestingly the extreme far end are enriched for receptors?

#https://www.uniprot.org/id-mapping/uniprotkb/bdf29ef6d065c51b4d0919f26f9ecdf452bf6eae/overview

annot %>% filter(set == "receptors") %>% View()

py_run_string("
import numpy as np

# Sample numpy array
data = np.random.rand(1, 214, 500)
")

# Access the Python variable in R
data_array <- py$data

# Convert the array to the desired tibble format
data_tibble <- as_tibble(t(array(data_array, dim = c(500, 214))))

# Check the result
dim(data_tibble) 


#####

library(dplyr)
library(tidyr)
library(tibble)

pickle <- import('pickle')
numpy <- import("numpy")


data_dict <- py_run_string("

import pickle
import numpy as np
with open('/scratch/gpfs/cmcwhite/activations/human_proteins_dir/EGFR_HUMAN.fasta.pkl', 'rb') as f:
    data = pickle.load(f)
    data = np.squeeze(data['aa_activations'], axis=0)
    print(data)
    
")

",389065 = V389064"
tibble(val = py$data[,389065]) %>% rowid_to_column() %>% pull(val) %>% sum()
  ggplot(aes(x  = rowid, y = val)) + 
  geom_point()

