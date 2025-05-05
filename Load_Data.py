import numpy as np
from Bio import SeqIO
import os

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

bni = {'a':0, 't':1, 'g':2, 'c':3}

nucleotide_indexes = {'a': bni['a'], 't':bni['t'], 'g':bni['g'], 'c':bni['c'], 'w': (bni['a'],bni['t']), 
                      'n': (bni['a'],bni['t'],bni['g'],bni['c']), 'y': (bni['t'], bni['c']), 's': (bni['g'], bni['c']), 
                      'm': (bni['a'],bni['c']), 'k': (bni['t'],bni['g']), 'r': (bni['a'], bni['g']), 'd': (bni['a'], bni['g'], bni['t'])}
nucleotide_values = {'a':1, 't':1, 'g':1, 'c':1, 'w': 0.5, 'n': 0.25, 'y': 0.5, 's': 0.5, 'm': 0.5, 'k': 0.5, 'r': 0.5, 'd':0.3}

class chromatin_dataset(Dataset):
    def __init__(self, xy):
        self.x_data=np.array([el[0] for el in xy],dtype=np.float32)
        self.y_data =np.array([el[1] for el in xy],dtype=np.float32)
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data)
        self.length=len(self.x_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length

def one_hot_encode(seq):
    oneHotEncode = np.zeros([4,len(seq)])
    for i, n in enumerate(seq):
        oneHotEncode[nucleotide_indexes[n.lower()], i] = nucleotide_values[n.lower()]
    return oneHotEncode

def load_data(data_dir:str, val_chr:str='Chr5', training_testing_split:float=0.7, 
              test_training_data_to_load:float=500, val_data_to_load:float=100):
    # get the .fasta files in data_dir
    fasta_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if 'fasta' in f]
    faste_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if 'faste' in f]


    # find the parts of the fasta files that is apart of val_chr
    X_chr = []
    with open(fasta_files[0], "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            char  = (record.id).split(',')[0]
            X_chr.append(char)

    testing_training_index = np.array([i for i, chr in enumerate(X_chr) if chr != val_chr])
    val_indices = np.array([i for i, chr in enumerate(X_chr) if chr == val_chr])

    testing_training_index = testing_training_index[:min(test_training_data_to_load, len(testing_training_index))]
    val_indices = val_indices[:min(val_data_to_load, len(val_indices))]

    # Load the the sequences
    X_val = []
    X_training_testing = []
    print(f'Loading sequences from {os.path.basename(fasta_files[0])}')
    with open(fasta_files[0], "rt") as handle:
        for i, record in enumerate(SeqIO.parse(handle, "fasta")):
            if i in val_indices:
                X_val.append(one_hot_encode(record.seq))
            if i in testing_training_index:
                X_training_testing.append(one_hot_encode(record.seq))
            if i > np.max(val_indices) and i > np.max(testing_training_index):
                break

    Y_val = []
    Y_training_testing = []
    for s in faste_files:
        print(f'Loading coverage from {os.path.basename(s)}')
        Y0 = []
        Y1 = []
        with open(s, "rt") as handle:
            for i, record in enumerate(SeqIO.parse(handle, "fasta")):
                if i in val_indices:
                    Y0.append(np.array([eval(str(s)) for s in record.seq.split(',')]))
                if i in testing_training_index:
                    Y1.append(np.array([eval(str(s)) for s in record.seq.split(',')]))
                if len(val_indices) > 0 and i > np.max(val_indices, 0):
                    break
        Y_val.append(Y0)
        Y_training_testing.append(Y1)
    
    # Reorder Y from (Tissue, Seq) -> (Seq, Tissue)
    Y_training_testing = np.array(Y_training_testing)
    Y_val = np.array(Y_val)
    Y_training_testing = np.swapaxes(Y_training_testing, 0, 1)
    Y_val = np.swapaxes(Y_val, 0, 1)

    X_val = np.array(X_val)
    X_training_testing = np.array(X_training_testing)

    # Discretize Y
    Y_training_testing = np.sum(Y_training_testing, axis=-1)
    Y_val = np.mean(Y_val, axis=-1)



    # Split the trainig and testing data
    x_train, x_test, y_train, y_test = train_test_split(
        X_training_testing, Y_training_testing, test_size=(1 - training_testing_split), random_state=42
    )

    # put the data into a dataloader
    training_dataset = chromatin_dataset([(x,y) for x,y in zip(x_train, y_train)])
    testing_dataset = chromatin_dataset([(x,y) for x,y in zip(x_test, y_test)])
    validation_dataset = chromatin_dataset([(x,y) for x,y in zip(X_val, Y_val)])

    return [training_dataset, testing_dataset, validation_dataset]


if __name__ == '__main__':
    training_dataset, testing_dataset, validation_dataset = load_data(os.path.join(os.getcwd(), 'Data', 'Parsed_Data'), test_training_data_to_load=10, val_data_to_load=10)
    print(len(training_dataset))
    print(len(testing_dataset))
    print(len(validation_dataset))
    print(training_dataset[0])











































