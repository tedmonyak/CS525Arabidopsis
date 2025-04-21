import numpy as np
from Bio import SeqIO
import os

import torch
from torch.utils.data import Dataset, DataLoader


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

def seqtopad(sequence, motif_len):
    rows=len(sequence)+2*motif_len-2
    S=np.empty([rows,4])
    base=['A', 'C', 'G', 'T']
    for i in range(rows):
        for j in range(4):
            if (i-motif_len+1<len(sequence) and sequence[i-motif_len+1]=='N' 
                or i<motif_len-1 or i>len(sequence)+motif_len-2):
                S[i,j]=np.float32(0.25)
            elif sequence[i-motif_len+1]==base[j]:
                S[i,j]=np.float32(1)
            else:
                S[i,j]=np.float32(0)
    return np.transpose(S)

def load_file(path, motif_len=24):
    dataset=[]
    sequences=[]
    with open(path, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            coverage = float((record.id).split(',')[3])
            sequence = str(record.seq)
            dataset.append([seqtopad(sequence, motif_len),[coverage]])
            sequences.append(sequence)
    return dataset  


def load_data(data_dir:str, val_chr:str='Chr5', testing_chr:str=None, training_testing_split:float=0.7):
    # get the .fasta files in data_dir
    fasta_files = [f for f in os.listdir(data_dir) if 'fasta' in f]
    x_train = []
    x_test = []
    x_val = []


    # get the .fastb files in data_dir
    fastb_files = [f for f in os.listdir(data_dir) if 'fastb' in f]
    y_train = []
    y_test = []
    y_val = []
    with open(path, "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):


    # put the data into a dataloader
    training_dataset = chromatin_dataset([(x,y) for x,y in zip(x_train, y_train)])
    testing_dataset = chromatin_dataset([(x,y) for x,y in zip(x_test, y_test)])
    validation_dataset = chromatin_dataset([(x,y) for x,y in zip(x_val, y_val)])

    return training_dataset, testing_dataset, validation_dataset















































