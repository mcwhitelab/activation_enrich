import argparse
import pickle
import blitzgsea as blitz
import pandas as pd
import numpy as np
from time import time
from concurrent.futures import ThreadPoolExecutor
from scipy.stats import hypergeom
import os
from numba import njit
from numba import config
from concurrent.futures import ProcessPoolExecutor


def parse_arguments():

    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Open a pickle file and print its content.")

    parser.add_argument("-p", "--pickle_file", type=str, required = True,  help="Path to the pickle file to open.")
    parser.add_argument("-s", "--seqname_file", type=str, required = True, help="Path to sequence  names, one per line, no header.")
    parser.add_argument("-c", "--conversion_file", type=str,  required = False, help="For converting between uniprot and entrez gene names the enrichr needs")

    parser.add_argument("-e", "--enrichr_pkl", type=str,  required = True, help="Path to downloaded enrichr dataset")
    parser.add_argument("-n", "--max_neurons", type=int, required = False,  help="Maximum number of neurons to run enrichments on")
    parser.add_argument("-f", "--fdr_threshold", type=float, required = False,  help="Only output neurons with at least this threshold fdr enrichment, default 0.01", default = 0.01)
    parser.add_argument("-ch", "--chunk_size", type=int, required = False,  default = 1000, help="Maximum number of neurons to run enrichments on")
    parser.add_argument("-pa", "--use_parallel",  action = "store_true",  help="Whether to use parallelization")
    parser.add_argument("-z", "--zscore_thresh", type=float, required = False, help="Threshold z score for activations, default 3", default = 3)
    parser.add_argument("-pv", "--threshold_p_value", type=float, required = False, help="Threshold p value for hypergeometric, default 1e-6", default = 1e-6)
    parser.add_argument("-sp", "--threshold_specificity", type=float, required = False, help="When a neuron is enriched for a term, neuron activators must be x * 100 percent annotated with term, default 0.8", default = 0.8)
    parser.add_argument("-l", "--lib_type", type = str, required = False, default = "genename", help = "If the enrichment library is in terms of gene names or uniprot entry, default = 'genename', other option is 'uniprot'")
    return parser.parse_args()

@njit

def binom(n, k):

    if k < 0 or k > n:

        return 0.0

    if k == 0 or k == n:

        return 1.0

    k = min(k, n - k)  # Take advantage of symmetry

    c = 1.0

    for i in range(k):

        c = c * (n - i) / (i + 1)

    return c

# Define the hypergeometric PMF

@njit

def hypergeom_pmf(k, M, n, N):

    return binom(n, k) * binom(M - n, N - k) / binom(M, N)

# Define the hypergeometric CDF

@njit
def hypergeom_cdf(k, M, n, N):

    cdf = 0.0

    for i in range(k + 1):

        pmf = hypergeom_pmf(i, M, n, N)

        cdf += pmf

    return cdf

# Define the function to calculate p-values

@njit
def calculate_p_values(A, G, M, threshold_p_value):

    results = []

    for i in range(A.shape[0]):

        for j in range(G.shape[0]):

            n = np.sum(G[j])  # Number of successes in G

            N = np.sum(A[i])  # Number of draws (successes in A)

            x = np.sum(A[i] * G[j])  # Number of observed successes

            p_value = 1 - hypergeom_cdf(x - 1, M, n, N)

            if p_value < threshold_p_value:

                results.append((i, j, p_value))

    return results

def process_chunk(start_index, chunk, G, M, threshold_p_value):
    # Calculate p-values for the chunk

    chunk_results = calculate_p_values(chunk, G, M, threshold_p_value)

    # Adjust the indices in the results to reflect their position in the original A matrix
    adjusted_results = [(i + start_index, j , p_val) for (i, j, p_val) in chunk_results]

    return adjusted_results

def parallel_p_value_calculation(A, G, M, threshold_p_value, num_chunks):

    # Split A into chunks
    chunk_size = A.shape[0] // num_chunks
    chunks = [(i, A[i:i + chunk_size]) for i in range(0, A.shape[0], chunk_size)]
    results = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_chunk, start_idx, chunk, G, M, threshold_p_value) for start_idx, chunk in chunks]
        for future in futures:
            results.extend(future.result())

    return results

def format_library(library_dict, index_list):
    """
    Adds new rows to the DataFrame based on a list of lists of gene names.
    Parameters:
    df (pandas.DataFrame): The original DataFrame.
    list_of_gene_lists (list of list of str): List of lists, each containing gene names.

    Returns:

    a presence/absence array for each gene list, matching the column order in the activation enrichment df
    a list of the genelist ids, 
    a list of lists of each genelist

    """


    out_rows = []
    for list_id, gene_list in library_dict.items():

        new_row = [1 if gene in gene_list else 0 for gene in index_list]

        out_rows.append(new_row)

    array = np.array(out_rows) 
    genelist_ids = list(library_dict.keys())
    genelists = list(library_dict.values())

    return array, genelist_ids, genelists


def split_activations(df):

     # went from 5 minutes -> 3 minutes -> 50 seconds with chatgpt suggestions for speeding up
     #df = pd.concat([df] * 5, ignore_index=True)

     array_split = np.hsplit(df.values, df.shape[1])
     
     # The next line take awhile because of need to create all the pandas dataframes
     # parallel slows down
      
     time_start = time()
     df_list = [pd.DataFrame({0: df.index, 1: part.squeeze()}) for part in array_split]


     # About 50 seconds for ~400000 activation column for 5 proteins 
     # Speed is more related to number of columns/neurons not the number of proteins
     #df_list = [pd.DataFrame(data=part, index=df.index, columns=[df.columns[i]]).reset_index().rename(columns = {'index':0, df.columns[i]:1}) for i, part in enumerate(array_split)]
     
     print("takes ", time() - time_start)

     #time_start = time()
     #def create_df(part):

     #   return pd.DataFrame({0: df.index, 1: part.squeeze()})



     #with ThreadPoolExecutor() as executor:

     #   df_list = list(executor.map(create_df, array_split))

     #print("takes ", time() - time_start)


    #print("pre array split") # This is very fast
     #df = pd.concat([df] * 5000, ignore_index=True)

     array_split = np.hsplit(df.values, df.shape[1])
     #print("array is split") 
     # Why does this next line take so long?
      
     #time_start = time()
     #df_list = [pd.DataFrame({0: df.index, 1: part.squeeze()}) for part in array_split]


     # About 50 seconds for ~400000 activation column for 5 proteins 
     # is speed related to columns or number of proteins?
     #df_list = [pd.DataFrame(data=part, index=df.index, columns=[df.columns[i]]).reset_index().rename(columns = {'index':0, df.columns[i]:1}) for i, part in enumerate(array_split)]
     
     #print("takes ", time() - time_start)





     return(df_list)


def split_activations_slower(df):
     #array_split = np.split(df.values, df.shape[1], axis=1)
 
     #df_list = [pd.DataFrame(data=part, index=df.index, columns=[df.columns[i]]) for i, part in enumerate(array_split)]


     print("pre array split") # This is very fast
     array_split = np.split(df.values, df.shape[1], axis=1)
     print("array is split") 
     # Why does this next line take so long?

     time_start = time()
     # About 3 minutes for activations for 5 proteins 
     # is speed related to columns or number of proteins?
     df_list = [pd.DataFrame(data=part, index=df.index, columns=[df.columns[i]]).reset_index().rename(columns = {'index':0, df.columns[i]:1}) for i, part in enumerate(array_split)]
     
     print("takes ", time() - time_start)


 

     df_list = [df[[col]].reset_index().rename(columns={'index': 0, col : 1}) for col in df.columns]
     # This is slowish like 3 minutes, since there are 393215 neuron in prot t5 xl
     print(df_list[1])
     print(len(df_list))
     #df_list = [pd.DataFrame({0: seqnames, 1: seq_activations[:, i]}) for i in range(1000)]
     # Actually do the gene name conversion here, otherwise all out of sync. 

     return(df_list)



def main():

    args = parse_arguments()

    pickle_file = args.pickle_file
    seqname_file = args.seqname_file
    conversion_file = args.conversion_file
    enrichr_pkl = args.enrichr_pkl
    chunk_size = args.chunk_size
    max_neurons = args.max_neurons
    use_parallel = args.use_parallel
    fdr_threshold = args.fdr_threshold
    zscore_thresh = args.zscore_thresh #3
    threshold_p_value = args.threshold_p_value #1e-6
    threshold_specificity = args.threshold_specificity # 0.8
    lib_type = args.lib_type
    # list available gene set libraries in Enrichr

    print("Input activations: {}".format(pickle_file))
    print("Input sequence names for activations: {}".format(seqname_file))
    print("Input gene id conversions: {}".format(conversion_file))
    print("Input enrichr set: {}".format(enrichr_pkl))


    enrichr_name = os.path.splitext(os.path.basename(enrichr_pkl))[0]


    dir_part=os.path.dirname(pickle_file)

    filename_wo_ext = os.path.splitext(os.path.basename(pickle_file))[0]
    output_filename = "output_" + filename_wo_ext   

    if dir_part:
        
        outdir = os.path.join(dir_part, output_filename)

    else:
        outdir = output_filename

    print("Making output directory {}".format(outdir))

    os.makedirs(outdir, exist_ok=True)
 
    #requireds internet
    #blitz.enrichr.print_libraries()
    # use enrichr submodule to retrieve gene set library

    # This is a dictionray of gene sets {id:[gene1, gene2]}
    with open (enrichr_pkl, 'rb') as f:
         library = pickle.load(f)


    #library = blitz.enrichr.get_library("GO_Molecular_Function_2017b")

    #example_signature = pd.read_csv("https://github.com/MaayanLab/blitzgsea/raw/main/testing/ageing_muscle_gtex.tsv")

    #print(example_signature)
    #example_result = blitz.gsea(example_signature, library)
    #print(example_result)
    
    #print(example_result[example_result['fdr'] < 0.05])
    #example_result.to_csv("tester.csv")
    ## run enrichment analysis
    #exit(1)


    with open(seqname_file, 'r') as f:
         seqnames_in = [line.strip() for line in f]
         #if conversion_file:
         seqnames = [x.split("|")[1] for x in seqnames_in] # accession
         seqnames_entryname = [x.split("|")[2] for x in seqnames_in] # entryname 
         acc_entryname = dict(zip(seqnames, seqnames_entryname))


        #print(conversion_dict)
         #seqnames = [conversion_dict[x.split("|")[1]] for x in seqnames if x in conversion_dict.keys()]
         #print(seqnames)

    with open(pickle_file, 'rb') as f:

        print("try to open")
        seq_activations = pickle.load(f)['sequence_activations']
        #print(seq_activations[1])
        #print(len(seq_activations[1]))
        # seq_activations is list of activation arrays. One per sequence
        neuron_idxs = list(range(0, seq_activations[1].shape[0]))  # OK, just going to index from zero for simplicity in matching to the aa activations
        #print(neuron_idxs[0:5])
        #print(len(neuron_idxs))
        #print(seq_activations.shape)

        if max_neurons:
            #seq_activations = seq_activations.iloc[:,-max_neurons:]
            print("try filter")
            seq_activations = [x[-max_neurons:] for x in seq_activations]
            print("done")
            neuron_idxs= neuron_idxs[-max_neurons:]
    

        print("convert to pd")
        seq_activations = pd.DataFrame(seq_activations, index = seqnames)
        

        #seq_activations.to_csv("small_sel.csv")
        print("pandas")

        try:
            assert len(seqnames) == len(seq_activations), "Number of seqeuence names ({}) must match number of rows of activations ({})".format(len(seqnames), len(seq_activations))

        except AssertionError as e:
            print(e)
            sys.exit(1)  

        print("Sequence activations")
        print(seq_activations)
        if conversion_file:
             # very specifically about converting uniprot sp|P00001|P00001_HUMAN to Entrez gene name for enrichr datasets
             conversion = pd.read_csv(conversion_file, sep='\t', header=None)
             #print(conversion)
             conversion.columns =["uniprot", "entrez"]    
             #print(conversion)
             conversion = conversion.dropna().set_index('uniprot')   #.to_dict()
             conversion.index.name = None
             
             print("Gene name conversion")
             print(conversion)
             seq_activations = conversion.join(seq_activations, how='inner')
             seq_activations.reset_index(drop = True, inplace=True)
             seq_activations.set_index('entrez', inplace = True)
             seq_activations.index.name = None
             
    print("Sequence activations", seq_activations)

    zscore = True
    if zscore == True:
        print("neuron_idxs", neuron_idxs)
        seq_activations.columns = neuron_idxs
        print(seq_activations)
        z_scores = (seq_activations - seq_activations.mean()) / seq_activations.std()
        print("got zscores")
        print(z_scores)
        if z_scores.columns[4] == neuron_idxs[4]:
               print("yes they're equal")
        else:
           print("nope")

        

        #def get_rows_greater_than_4(col_name_and_data):
        #    col_name, column_data = col_name_and_data
        #    return {col_name: list(column_data[column_data > 4].index)}
        #
        #with ThreadPoolExecutor() as executor:
        #    results = list(executor.map(get_rows_greater_than_4, [(col, z_scores[col]) for col in z_scores.columns]))
        # 
        #results_dict = {k: v for d in results for k, v in d.items()}
        
        # really only need entries for things above threshold 
        results_dict = {}
        def update_rows_greater_than_4(col_name_and_data):
                col_name, column_data = col_name_and_data
                value = list(column_data[column_data > 4].index)
                if len(value) > 0:
                    results_dict[col_name] = value
    
        with ThreadPoolExecutor() as executor:
                 executor.map(update_rows_greater_than_4, [(col, z_scores[col]) for col in z_scores.columns])

        numba = True

        zscore_thresh = 3
        threshold_p_value = 1e-6
        outfile = os.path.join(outdir, f"{filename_wo_ext}.lib{enrichr_name}.z{zscore_thresh}.p{threshold_p_value}.txt")

        if numba == True:

            with open(outfile, "w") as o:
               o.write("neuron_name,in_geneset_prop,coverage,pvalue,neuron_activators,geneset_names,eneset_id,neuron_activator_size, geneset_size,overlap_size,not_in_geneset\n")
               A = (z_scores > zscore_thresh).astype(int).values.T
               zscores_index = z_scores.index.tolist()  # these are the genenames
               G, library_ids, library_genelists = format_library(library, zscores_index)
   
               print(A)
               print(library_genelists[0:5])
 
               print(A.shape)
               print(G.shape)
               M = A.shape[1]
    
               time_parallel = time()
               p_values = parallel_p_value_calculation(A, G, M, threshold_p_value, num_chunks=10)
               print("ptime", time() - time_parallel)
   
               for neuron_index, genelist_index, pvalue in p_values:
                   neuron_name =  z_scores.columns.tolist()[neuron_index]
                   geneset_id = library_ids[genelist_index]
                   neuron_activators = results_dict[z_scores.columns.tolist()[neuron_index]]
                   geneset_names = library_genelists[genelist_index]
                   neuron_activator_size = len(neuron_activators)
                   geneset_size = len(geneset_names)
                   #coverage = neuron_activator_size / geneset_size
                   overlap_size = len(set(neuron_activators).intersection(set(geneset_names)))  
                   coverage = overlap_size / geneset_size 
                   not_in_geneset = neuron_activator_size - overlap_size#/ neuron_activator_size
                   not_in_geneset_prop = not_in_geneset / neuron_activator_size
                   in_geneset_prop =  overlap_size/neuron_activator_size

                   if in_geneset_prop >= threshold_specificity:
                       if conversion_file:
                            geneset_string   = ' '.join(geneset_names)
                            neuron_string  = ' '.join(neuron_activators)
                       else: 
                            geneset_string = " ".join([acc_entryname[x] for x in geneset_names])                          
                            neuron_string = " ".join([acc_entryname[x] for x in neuron_activators])
                       o.write(f"{neuron_name},{in_geneset_prop},{coverage},{pvalue},{neuron_string},{geneset_string},{geneset_id},{neuron_activator_size},{ geneset_size},{overlap_size},{not_in_geneset}\n")
                   # so maximize overlap and coveraget
                   else:
                       continue
                   print("significant", pvalue)
                   print("neuron name", z_scores.columns.tolist()[neuron_index])
                   print("geneset id", library_ids[genelist_index])
                   print("neuron activators", " ".join(neuron_activators))
                   print("geneset names    ", " ".join(geneset_names))
                   print("overlap", overlap_size)
                   print("geneset_size, activator_size", geneset_size, neuron_activator_size)
                   print("coverage prop", coverage)
                   print("not in geneset prop", not_in_geneset_prop) 
   
               #print(p_values[-5:])

            exit(1)
 

        else:
    
            print(results_dict)
            universe = z_scores.index.tolist()
            #print(universe)
            M = len(universe)
            #print(M)
            #exit(1)
    
            for neuron_id, geneset in results_dict.items():
                print(neuron_id)
                n = len(geneset)
    
                print("num_categories", len(library.keys()))
                for refid, refgenes in library.items():
                    #print(refid)
                    time_intersect = time()
                    N = len(refgenes)
                    x = len(list(set(geneset).intersection(set(refgenes))))
                    print("intersect", time() - time_intersect)
      
                    time_hypergeom = time()
                    p_value_real = 1 - hypergeom.cdf(x-1, M, n, N)
                    print("hypergeom", time() - time_hypergeom)
    
    
                    # Simulations
                    time_sim = time()
                    simulated_p_values = []
               
                    refgene_set = set(refgenes)
                    for _ in range(10):
                         simulated_list = set(np.random.choice(list(universe), n, replace=False))
                         simulated_overlap = len(simulated_list.intersection(refgene_set))
                         p_value_simulated = 1 - hypergeom.cdf(simulated_overlap-1, M, n, N)
                         simulated_p_values.append(p_value_simulated) 
               
                    print("simtime", time() - time_sim)
                    percentile = 100 * sum(p <= p_value_real for p in simulated_p_values) / len(simulated_p_values)
                    
    
                    # 3. Compute the Proportion of Simulated p-values Less Than the True p-value
                    empirical_p_value = sum(p <= p_value_real for p in simulated_p_values) / len(simulated_p_values)
                    if p_value_real < 0.0000005:
    
                        print("refid", refid)  
                        print("geneset", " ".join(geneset))
                        print("refgenes", " ".join(refgenes))
                        print(f"True p-value {p_value_real} is at the {percentile:.15f} percentile of the simulated p-values.")                 
                        print(f"Empirical p-value: {empirical_p_value:.12f}")


        print(results)
        with open('results.pkl', 'wb') as f:

               pickle.dump(results, f)

    exit(1)

    if gsea == True:

        start1 = time()
        np.random.seed(42)
        # indexing neurons from zero...
        signatures = split_activations(seq_activations)
        func1time = time() - start1
        print(func1time)
    
        if use_parallel == True:
            starttime = time()
            def process_chunk(chunk):
        
                for idx, signature in chunk:
            
                    print(idx)
            
                    result = blitz.gsea(signature, library, shared_null=True)
            
                    float_cols = result.select_dtypes(include=['float64']).columns
            
                    result[float_cols] = result[float_cols].round(4)
            
                    output = result[result['fdr'] < fdr_threshold]
            
                    if len(output) > 0:
            
                        print("writing {}".format(idx))
            
                        outfile = os.path.join(outdir, f"{filename_wo_ext}.neuron{idx}.lib{enrichr_name}.thresh{fdr_threshold}")
            
                        output.to_csv(outfile)
        
            starttime = time()
        
            # TODO, can this be parallelized?
            print("signatures prepared")
            
            def chunks(lst, n):
        
                """Yield successive n-sized chunks from lst."""
        
                for i in range(0, len(lst), n):
        
                    yield lst[i:i + n]
        
            with ThreadPoolExecutor() as executor:
         
                #list(executor.map(process_chunk, chunks(list(enumerate(signatures)), chunk_size)))
                list(executor.map(process_chunk, chunks(list(zip(neuron_idxs[::-1], signatures[::-1])), chunk_size)))
            print("parallel time", time() - starttime)
    
        else:
            starttime = time()
            for idx, signature in zip(neuron_idxs[::-1], signatures[::-1]):
                  idx_time = time()
                  print(idx)
                  result = blitz.gsea(signature, library, shared_null= True)
                  float_cols = result.select_dtypes(include=['float64']).columns
        
                  result[float_cols] = result[float_cols].round(4)
                  output  = result[result['fdr'] < fdr_threshold]
                  if len(output) > 0:
                      print("writing {}".format(idx))
                      outfile = os.path.join(outdir, f"{filename_wo_ext}.neuron{idx}.lib{enrichr_name}.thresh{fdr_threshold}.noparallel")
                      output.to_csv(outfile)
                  print("idx time", time() - idx_time)
            print("no parallel time", time() - starttime)

if __name__ == '__main__':

    main()


