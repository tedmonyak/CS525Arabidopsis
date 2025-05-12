import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepCNN(nn.Module):
    def __init__(self, num_kernels=[128, 64, 32, 16], kernel_size=[10,10,10,10],
                 dropout=0, output_size=37, flattened_size=0, fc_size=[120, 84]):
        super(DeepCNN, self).__init__()
        self.input_channels=4
        self.num_kernels=num_kernels
        self.kernel_size=kernel_size
        self.dropout=dropout
        self.conv1 = nn.Conv1d(4, num_kernels[0], kernel_size[0])
        self.pool = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(num_kernels[0], num_kernels[1], kernel_size[1])
        self.conv3 = nn.Conv1d(num_kernels[1], num_kernels[2], kernel_size[2])
        self.fc1 = nn.Linear(flattened_size, fc_size[0]) 
        self.fc2 = nn.Linear(fc_size[0], fc_size[1])
        self.fc3 = nn.Linear(fc_size[1], output_size)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x