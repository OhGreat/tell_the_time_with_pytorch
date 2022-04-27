import numpy as np
import torch
import time
import math
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, random_split
from classes.ClockDataset import ClockDataset
from classes.CommonSenseError import CommonSenseError
from classes.Models import NN_regression
from utilities import *


def main():
    # main modality
    mode = "periodic_labels"
    # data parameters
    data_splits = [16500,1000,500]
    batch_size = 64
    input_channels = 1
    img_height = 150
    img_width = 150
    # model parameters
    n_outputs = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NN_regression(  input_channels=input_channels,
                            h=img_height,w=img_width,
                            n_outputs=n_outputs).to(device)
    if mode == "periodic_labels":
        loss = nn.MSELoss()
    else:
        print("Please choose a correct mode.")
        exit()
    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    save_weights = "periodic_labels_2"
    save_losses = True
    epochs = 150
    patience = 10
    # extra parameters
    verbose = 2

    if verbose > 0:
        print(f"Running pytrch on {device} device.\n")

    # Load data
    data, labels = load_data()
    if verbose > 0:
        print(f"data shape: {data.shape}, labels shape: {labels.shape}")

    # Transform labels to periodic values if needed.
    if mode == "periodic_labels":
        labels = transform_labels(labels)

    # Create main dataset
    clock_dataset = ClockDataset(data, labels)
    # Split dataset into train, test and validation sets
    train_data, val_data, test_data = random_split(clock_dataset, data_splits)
    # Create data loaders
    train_data_loader = DataLoader(train_data, batch_size=batch_size)
    val_data_loader = DataLoader(val_data, batch_size=batch_size)
    test_data_loader = DataLoader(test_data, batch_size=batch_size)

    # Prepare target array to use for evaluation
    targets = []
    for i in range(len(test_data)):
        targets.append(test_data[i][1])
    targets = np.vstack(targets)
    if verbose > 1:
        for X, y in train_data_loader:
            print(f"Shape of data batch [N, C, H, W]: {X.shape}")
            print(f"Shape of target batch: {y.shape}")
            break

    # Main training loop
    curr_patience = 0
    train_losses = []
    eval_losses = []
    mean_test_loss = np.inf
    print("\n~~~Starting training~~~\n")
    start_time = time.time()
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        # Training step
        train_l = train(train_data_loader, model, loss, optimizer, device)
        train_mean = torch.mean(train_l)
        train_losses.append(train_mean)
        print(f"Train loss: {train_mean:>7f}")

        # Evaluation step
        eval_l = evaluate(val_data_loader, model, loss, device)
        eval_mean = torch.mean(eval_l)
        eval_losses.append(eval_mean)
        print(f"Test avg loss: {eval_mean:>8f}")

        # Save new weights if they are better
        if (save_weights != None) and (eval_mean < mean_test_loss):
            print("Found new best weights.")
            mean_test_loss = eval_mean
            torch.save(model.state_dict(), "model_weights/"+save_weights)
            curr_patience = 0
        print()

        # Stop training if patience expired
        if curr_patience >= patience:
            print(f"No new best weights found after {patience} iterations.")
            break
        curr_patience += 1

    end_time = time.time()
    print(f"Training finished in {end_time-start_time} seconds.")

    # make predictions on test dataset
    predictions = predict(test_data_loader, model, loss, device)
    # transform cosine and sine back to integer values
    true_preds = denormalize_time(predictions)
    # calculate and print common sense error
    cse = CommonSenseError()
    cse_error = cse(torch.FloatTensor(true_preds),torch.FloatTensor(targets[:,:2]))
    print(f"Common sense error on test dataset: {np.round(cse_error.numpy(),3)}")

    if save_losses:
        train_losses = np.vstack(train_losses)
        test_losses = np.vstack(eval_losses)
        save_train_plot(train_losses, test_losses, save_weights)

def train(dataloader, model, loss_fn, optimizer, device):
    """ Applies backpropagation to train the model
    """
    losses = []
    model.train()
    for X, y in dataloader:
        X, y = X.to(device), y[:, 2:].to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss)
    losses = torch.FloatTensor(losses)
    return losses

def evaluate(dataloader, model, loss_fn, device):
    """ Used to evaluate the model on unknown data
        during training 
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y[:, 2:].to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            losses.append(loss)
    losses = torch.FloatTensor(losses)
    return losses


def predict(dataloader, model, loss_fn, device):
    """ Returns predictions for the data in the DataLoader 
        as one single batch.
    """
    losses = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y[:,2:].to(device)
            pred = model(X)
            predictions.append(pred.cpu().numpy())
            loss = loss_fn(pred, y)
            losses.append(loss)
    losses = torch.FloatTensor(losses)
    print(f"Avg evaluation loss: {torch.mean(losses):>8f} \n")
    # Normalize values bigger/smaller than the max/min possible
    predictions  = np.vstack(predictions)
    predictions[ predictions > 1 ] = 1
    predictions[ predictions < -1 ] = -1
    return predictions

def denormalize_time(predictions):
    """ Returns the corresponding hour and minute
        for each entry of the predicted sine and cosine values.
    """
    true_preds = []
    for h_cos,h_sin,m_cos,m_sin in predictions:
        h_angle = math.atan2(h_sin, h_cos)
        h_angle *= 180 / math.pi
        if h_angle < 0: h_angle += 360

        m_angle = math.atan2(m_sin, m_cos)
        m_angle *= 180 / math.pi
        if m_angle < 0: m_angle += 360

        true_hour = math.modf(h_angle/30)[1]
        true_mins = math.modf(m_angle/6)[1]
        true_preds.append([true_hour, true_mins])
    return torch.FloatTensor(true_preds)

def save_train_plot(train_loss, eval_loss, plot_name):
    """ Function to plot the losses during training and 
        save the figure.
    """
    plt.plot(train_loss, label="train")
    plt.plot(eval_loss, label="evaluate")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig("train_plots/"+plot_name+".png")

if __name__ == "__main__":
    main()