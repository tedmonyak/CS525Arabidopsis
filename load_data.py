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

def load_data(data_dir:str, test_chr:str='Chr5', train_val_split:float=0.8, 
              train_val_data_to_load:float=500, test_data_to_load:float=100):
    # get the .fasta files in data_dir
    fasta_files = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if 'fasta' in f]
    faste_files = [os.path.join(data_dir,f) for f in os.listdir(data_dir) if 'faste' in f]

    # find the parts of the fasta files that is apart of val_chr
    X_chr = []
    with open(fasta_files[0], "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            char  = (record.id).split(',')[0]
            X_chr.append(char)

    train_val_indices = np.array([i for i, chr in enumerate(X_chr) if chr != test_chr])
    test_indices = np.array([i for i, chr in enumerate(X_chr) if chr == test_chr])

    train_val_indices = train_val_indices[:min(train_val_data_to_load, len(train_val_indices))]
    test_indices = test_indices[:min(test_data_to_load, len(test_indices))]


    # Load the the sequences
    X_test = []
    with open(fasta_files[0], "rt") as handle:
        for i, record in enumerate(SeqIO.parse(handle, "fasta")):
            if i in test_indices:
                X_test.append(one_hot_encode(record.seq))

    X_train_val = []
    with open(fasta_files[0], "rt") as handle:
        for i, record in enumerate(SeqIO.parse(handle, "fasta")):
            if i in train_val_indices:
                X_train_val.append(one_hot_encode(record.seq))

    Y_test = []
    with open(faste_files[0], "rt") as handle:
        for i, record in enumerate(SeqIO.parse(handle, "fasta")):
            if i in test_indices:
                Y_test.append(np.array([eval(str(s)) for s in record.seq.split(',')]))

    Y_train_val = []
    with open(faste_files[0], "rt") as handle:
        for i, record in enumerate(SeqIO.parse(handle, "fasta")):
            if i in train_val_indices:
                Y_train_val.append(np.array([eval(str(s)) for s in record.seq.split(',')]))
    

    # Split the trainig and testing data
    x_train, x_val, y_train, y_val = train_test_split(
        X_train_val, Y_train_val, test_size=(1 - train_val_split), random_state=42
    )


    # put the data into a dataloader
    training_dataset = chromatin_dataset([(x,y) for x,y in zip(x_train, y_train)])
    validation_dataset = chromatin_dataset([(x,y) for x,y in zip(x_val, y_val)])
    testing_dataset = chromatin_dataset([(x,y) for x,y in zip(X_test, Y_test)])

    return training_dataset, validation_dataset, testing_dataset


if __name__ == '__main__':
    cwd = os.getcwd()
    training_dataset, validation_dataset, testing_dataset = load_data(cwd)
    print(len(training_dataset))
    print(len(validation_dataset))
    print(len(testing_dataset))
    print(training_dataset[0])











































