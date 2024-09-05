import requests

import pickle



def parse_data(file_path):

    prosite_dict = {}

    with open(file_path, 'r') as file:

        next(file)  # Skip the header line

        for line in file:

            parts = line.strip().split('\t')
            entry = parts[0]
            prosites = parts[1] if len(parts) > 1 else ''

            if prosites:
                for prosite in prosites.split(';'):

                    if prosite:
                        if prosite in prosite_dict:
                            prosite_dict[prosite].append(entry)

                        else:

                            prosite_dict[prosite] = [entry]

    return prosite_dict



def main():

    file_path = 'uniprot_data.tsv'  # Path to your downloaded TSV file

    data = parse_data(file_path)

    pickle_file_path = 'prosite_annotations.pkl'

    with open(pickle_file_path, 'wb') as pickle_file:

        pickle.dump(data, pickle_file)


    print(data)
    print(f"File saved as {pickle_file_path}")



if __name__ == '__main__':

    main()


