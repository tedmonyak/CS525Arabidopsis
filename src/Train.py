import os
import math
import torch
import numpy as np
from IPython.display import clear_output
import matplotlib.pyplot as plt
import json



def train_epoch(dataloader, model, loss_fn, optimizer, epoch, device):
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

def validation(dataloader, model, loss_fn, epoch, device):
    # set the model to evaluation mode 
    model.eval()
    # size of dataset
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    validation_loss = 0
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

def train_model(device, train_loader, val_loader, model, optimizer, loss_fn, epochs, save_dir, patience=10):
    def plot_loss_live(train_loss, validation_loss):
        clear_output(wait=True)
        plt.figure(figsize=(4,3))
        plt.plot(np.arange(len(train_loss)), train_loss, label='Training')
        plt.plot(np.arange(len(validation_loss)), validation_loss, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plot_save_path = os.path.join(save_dir, f"loss_plot_epoch_{len(train_loss)}.png")
        # plt.savefig(plot_save_path)
        plt.show()
        # plt.close()

    p = patience

    train_loss = []
    validation_loss = []
    best_loss = math.inf
    for t in range(epochs):
        if t % 10 == 0 :
            print(f"Epoch {t}\n-------------------------------")
            # Save train_loss and validation_loss to a JSON file
            losses = {
                "train_loss": train_loss,
                "validation_loss": validation_loss
            }
            save_path = os.path.join(save_dir, "losses.json")
            with open(save_path, "w") as f:
                json.dump(losses, f)
            torch.save(model, os.path.join(save_dir, 'model.pth'))

        loss = train_epoch(train_loader, model, loss_fn, optimizer, t, device)
        train_loss.append(loss)
        loss = validation(val_loader, model, loss_fn, t, device)
        validation_loss.append(loss)
        plot_loss_live(train_loss, validation_loss)

    
        if train_loss[-1] < validation_loss[-1]:
            # print(f"Training loss {train_loss[-1]} is less than validation loss {validation_loss[-1]}")
            if train_loss[-1]/validation_loss[-1] < 0.8:
                print(f"Training loss {train_loss[-1]} is less than half of validation loss {validation_loss[-1]}")
                p -= 1
        
        if p == 0:
            print(f"Early stopping at epoch {t}")
            break
                
    print("Done!")

