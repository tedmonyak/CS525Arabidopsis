import numpy as np
import os
import torch
import math

from Bio import SeqIO

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

def load_data(data_dir:str, test_chr:str='Chr5', train_val_split:float=0.7, 
              train_val_data_to_load:float=500, test_data_to_load:float=100,
              faste_files_to_load=37, normalize=False, upper_threshold=False, lower_threshold=False,
              minimum_coverage=0, maximum_coverage=1_000_000):
    # get the .fasta files in data_dir
    fasta_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if 'fasta' in f]
    faste_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if 'faste' in f]


    # find the parts of the fasta files that is a part of test_chr
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
    X_train_val = []
    print(f'Loading sequences from {os.path.basename(fasta_files[0])}')
    with open(fasta_files[0], "rt") as handle:
        for i, record in enumerate(SeqIO.parse(handle, "fasta")):
            if i in test_indices:
                X_test.append(one_hot_encode(record.seq))
            if i in train_val_indices:
                X_train_val.append(one_hot_encode(record.seq))
            if i > np.max(test_indices) and i > np.max(train_val_indices):
                break

    Y_test = []
    Y_train_val = []
    for f in faste_files[:faste_files_to_load]:
        print(f'Loading coverage from {os.path.basename(f)}')
        Y0 = []
        Y1 = []
        with open(f, "rt") as handle:
            for i, record in enumerate(SeqIO.parse(handle, "fasta")):
                if i in test_indices:
                    try:
                        v = math.log(float(str(record.seq))+1)
                    except ValueError:
                        print(f"Error parsing sequence in {f}, values for {i} are set to 0")
                        v = 0
                    Y0.append(v)
                if i in train_val_indices:
                    try:
                        v = math.log(float(str(record.seq))+1)
                    except ValueError:
                        print(f"Error parsing sequence in {f}, values for {i} are set to 0")
                        v = 0
                    Y1.append(v)
                if len(test_indices) > 0 and i > np.max(test_indices, 0):
                    break
        Y_test.append(Y0)
        Y_train_val.append(Y1)
    
    # Reorder Y from (Tissue, Seq) -> (Seq, Tissue)
    Y_train_val = np.array(Y_train_val)
    Y_test = np.array(Y_test)
    Y_train_val = np.swapaxes(Y_train_val, 0, 1)
    Y_test = np.swapaxes(Y_test, 0, 1)

    # threshold
    if upper_threshold:
        mean_train_val = np.mean(Y_train_val, axis=0)
        std_train_val = np.std(Y_train_val, axis=0)
        if maximum_coverage is None:
            maximum_coverage = mean_train_val + 1 * std_train_val
        threshold_train_val = maximum_coverage
        Y_train_val = np.clip(Y_train_val, None, threshold_train_val)
        Y_test = np.clip(Y_test, None, threshold_train_val)


    if lower_threshold:
        mean_train_val = np.mean(Y_train_val, axis=0)
        std_train_val = np.std(Y_train_val, axis=0)
        if minimum_coverage is None:
            minimum_coverage = mean_train_val - 2 * std_train_val
        threshold_train_val = minimum_coverage
        Y_train_val = np.clip(Y_train_val, threshold_train_val, None)
        Y_test = np.clip(Y_test, threshold_train_val, None)

    X_test = np.array(X_test)
    X_train_val = np.array(X_train_val)

    # Discretize Y
    # Y_train_val = np.sum(Y_train_val, axis=-1)
    # Y_test = np.sum(Y_test, axis=-1)

    if normalize:
        max_val = max(np.max(Y_test), np.max(Y_train_val))
        Y_test /= max_val
        Y_train_val /= max_val

    # Split the trainig and testing data
    x_train, x_val, y_train, y_val = train_test_split(
        X_train_val, Y_train_val, test_size=(1 - train_val_split), random_state=42
    )

    # put the data into a dataloader
    training_dataset = chromatin_dataset([(x,y) for x,y in zip(x_train, y_train)])
    validation_dataset = chromatin_dataset([(x,y) for x,y in zip(x_val, y_val)])
    testing_dataset = chromatin_dataset([(x,y) for x,y in zip(X_test, Y_test)])
    print("Done Loading Data")

    return [training_dataset, validation_dataset, testing_dataset]


def get_data_loaders(data_dir, batch_size=256, faste_files_to_load=37, normalize=False, train_val_data_to_load=math.inf, test_data_to_load=math.inf, upper_threshold=False, lower_threshold=False, minimum_coverage=None, maximum_coverage=None):
    Data = load_data(data_dir, 
                        train_val_data_to_load=train_val_data_to_load,
                        test_data_to_load=test_data_to_load,
                        faste_files_to_load=faste_files_to_load,
                        normalize=normalize,
                        upper_threshold=upper_threshold, 
                        lower_threshold=lower_threshold,
                        minimum_coverage=minimum_coverage,
                        maximum_coverage=maximum_coverage
                     )
    
    training_dataset, validation_dataset, testing_dataset = Data

    train_loader = DataLoader(dataset=training_dataset,
                              batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(dataset=validation_dataset,
                              batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(dataset=testing_dataset,
                              batch_size=batch_size,shuffle=True)
    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    training_dataset, testing_dataset, validation_dataset = load_data(os.path.join(os.getcwd(), 'Data', 'Parsed_Data'), train_val_data_to_load=10, test_data_to_load=10)
    print(len(training_dataset))
    print(len(validation_dataset))
    print(len(testing_dataset))
    print(training_dataset[0])
    print(training_dataset[1])












































