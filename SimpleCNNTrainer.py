# %%
import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import math
import matplotlib.pyplot as plt
import numpy as np

from Load_Data import load_data

device = (
    "mps"
    if torch.backends.mps.is_available()
    else "cuda"
    if torch.cuda.is_available()
    else "cpu"
)
device = torch.device(device)
print(f"Using {device} device")

# %% [markdown]
# # Load Data

# %%
def get_data_loaders(batch_size=64):
    Data = load_data(os.path.join(os.getcwd(), 'Data', 'Parsed_Data'), 
                        train_val_data_to_load=math.inf, 
                        test_data_to_load=math.inf,
                        faste_files_to_load=37
                        )
    
    training_dataset, validation_dataset, testing_dataset = Data

    train_loader = DataLoader(dataset=training_dataset,
                              batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(dataset=validation_dataset,
                              batch_size=batch_size,shuffle=True)
    test_loader = DataLoader(dataset=testing_dataset,
                              batch_size=batch_size,shuffle=True)
    return train_loader, val_loader, test_loader

train_loader, val_loader, test_loader = get_data_loaders()

# %% [markdown]
# # Build Model

# %%
class SimpleCNN(nn.Module):
    def __init__(self, num_kernels=[2048, 512, 128], kernel_size=[128,64,32],
                 dropout=0):
        super(SimpleCNN, self).__init__()
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
        # self.conv_block.append(nn.Sequential(
        #     nn.Conv1d(in_channels=self.num_kernels[0],
        #               out_channels=num_kernels[1],
        #               kernel_size=kernel_size[1]),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2),
        #     nn.Dropout(p=self.dropout),            
        # ))
        # Add a third convolutional layer
        # self.conv_block.append(nn.Sequential(
        #     # second layer
        #     nn.Conv1d(in_channels=self.num_kernels[1],
        #               out_channels=num_kernels[2],
        #               kernel_size=kernel_size[2]),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2),
        #     nn.Dropout(p=self.dropout),            
        # ))
        self.regression_block = nn.Sequential(
            nn.Linear(num_kernels[0], 37),
            nn.ReLU(),  # ReLU ensures positive outputs
            # nn.LogSoftmax(dim=1)  # Apply log softmax if necessary for your task
        )  

    def forward(self, x):
        x = self.conv_block(x)
        x,_ = torch.max(x, dim=2)        
        x = self.regression_block(x)
        return x

# %% [markdown]
# # Train Model

# %% [markdown]
# ### Training functions

# %%
def train_epoch(dataloader, model, loss_fn, optimizer, epoch):
    size = len(dataloader)
    num_batches = len(dataloader)
    total_loss = 0
    # set the model to training mode - important when you have 
    # batch normalization and dropout layers
    model.train()
    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        # Compute prediction and loss
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        # backpropagation
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 10 == 0 :
        print(f"training loss: {total_loss/num_batches:>7f}")
    return total_loss / num_batches

def validation(dataloader, model, loss_fn, epoch):
    # set the model to evaluation mode 
    model.eval()
    # size of dataset
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    validation_loss, correct = 0, 0
    # Evaluating the model with torch.no_grad() ensures that no gradients 
    # are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage 
    # for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            validation_loss += loss_fn(y_pred, y).item()
    validation_loss /= num_batches
    if epoch%10 == 0 :
        print(f"Validation Loss: {validation_loss:>8f} \n")
    return validation_loss

def train_model(train_loader, val_loader, model, optimizer):
    epochs = 1000
    loss_fn = nn.PoissonNLLLoss(log_input=True, full=True)
    patience = math.inf
    p = 100
    
    
    train_loss = []
    validation_loss = []
    best_loss = math.inf
    for t in range(epochs):
        if t % 10 == 0 :
            print(f"Epoch {t}\n-------------------------------")
        loss = train_epoch(train_loader, model, loss_fn, optimizer, t)
        train_loss.append(loss)
        loss = validation(val_loader, model, loss_fn, t)
        validation_loss.append(loss)
    
        if loss < best_loss:
            best_loss = loss    
            p = patience
        else:
            p -= 1
            if p == 0:
                print("Early Stopping!")
                break    
    print("Done!")

    def plot_loss(train_loss, validation_loss):
        plt.figure(figsize=(4,3))
        plt.plot(np.arange(len(train_loss)), train_loss, label='Training')
        plt.plot(np.arange(len(validation_loss)), validation_loss, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('training_validation_loss.png')
        plt.show()
    plot_loss(train_loss, validation_loss)


# %% [markdown]
# ### Train

# %%
model = SimpleCNN().to(device)
lr = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
train_model(train_loader, val_loader, model, optimizer)

# %% [markdown]
# ### Save Model

# %%
torch.save(model, "model.pth")

# %% [markdown]
# ### Train Model

# %%
model = torch.load("model.pth", weights_only=False)
model.to(device)
model.eval()

# %%
input, y = next(iter(train_loader))
input = input.to(device)

output = model.forward(input)
print(output.shape)

# %%
i = 8
print('Tissue: Predicted, True')
for s, (y_p, y_t) in enumerate(zip(output[i], y[i])):
    print(f'{s}: {torch.exp(y_p):.1f}, {y_t:.1f}')


# %%



