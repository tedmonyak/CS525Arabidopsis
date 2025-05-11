import torch.nn as nn
import torch


class DeepCNN(nn.Module):
    def __init__(self, num_kernels=[128, 64, 32, 16], kernel_size=[10,10,10,10],
                 dropout=0, output_size=37):
        super(DeepCNN, self).__init__()
        self.input_channels=4
        self.num_kernels=num_kernels
        self.kernel_size=kernel_size
        self.dropout=dropout
        self.conv_block = nn.Sequential(
            # first layer
            nn.Conv1d(in_channels=self.input_channels,
                      out_channels=num_kernels[0],
                      kernel_size=kernel_size[0]),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.MaxPool1d(kernel_size=2),
        )
        # second layer
        self.conv_block.append(nn.Sequential(
            nn.Conv1d(in_channels=self.num_kernels[0],
                      out_channels=num_kernels[1],
                      kernel_size=kernel_size[1]),
            #nn.BatchNorm1d(num_features=num_kernels[1]),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),   
            nn.MaxPool1d(kernel_size=2),        
        ))
        # Add a third convolutional layer
        self.conv_block.append(nn.Sequential(
            # second layer
            nn.Conv1d(in_channels=self.num_kernels[1],
                      out_channels=num_kernels[2],
                      kernel_size=kernel_size[2]),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),   
        ))
        self.regression_block = nn.Sequential(
            nn.Linear(num_kernels[2], output_size),
            nn.ReLU(),  # ReLU ensures positive outputs
        ) 

    def forward(self, x):
        x = self.conv_block(x)
        x,_ = torch.max(x, dim=2)     
        x = self.regression_block(x)
        return x