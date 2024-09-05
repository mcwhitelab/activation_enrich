import argparse
import pandas as pd
import os
import pickle



def process_pkl_file(file_path):

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

        

    # Creating a dictionary with the pathway names and their associated list lengths
    result = {key: len(value) for key, value in data.items()}

    

    # Convert the dictionary to a DataFrame for easy saving to a .csv file
    df = pd.DataFrame(list(result.items()), columns=['Pathway', 'Count'])

    

    # Saving to a .csv file
    output_path = os.path.splitext(file_path)[0] + "_counts.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved counts to {output_path}")



def main():

    parser = argparse.ArgumentParser(description="Process a set of .pkl files containing dictionaries.")
    parser.add_argument('files', metavar='F', type=str, nargs='+', help='a path to a .pkl file')

    args = parser.parse_args()



    for file_path in args.files:
        process_pkl_file(file_path)



if __name__ == '__main__':

    main()


