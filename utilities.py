import numpy as np
import pandas as pd
import torch
import math
import matplotlib.pyplot as plt

def load_data():
    """ Returns the data and labels as pytorch tensors.
    """
    data = np.load("data/data_1.npy")
    labels = np.load("data/labels_1.npy")

    data = np.vstack((data,np.load("data/data_2.npy")))
    labels = np.vstack((labels,np.load("data/labels_2.npy")))

    data = np.vstack((data,np.load("data/data_3.npy")))
    labels = np.vstack((labels,np.load("data/labels_3.npy")))

    data = np.vstack((data,np.load("data/data_4.npy")))
    labels = np.vstack((labels,np.load("data/labels_4.npy")))

    data = np.vstack((data,np.load("data/data_5.npy")))
    labels = np.vstack((labels,np.load("data/labels_5.npy")))

    #data = np.load("data/images.npy")
    data = torch.FloatTensor(np.expand_dims(data,axis=1))/255.
    data_permuts = torch.randperm(data.shape[0])
    data = data[data_permuts,:]
    labels = torch.FloatTensor(np.load("data/labels.npy"))
    labels = labels[data_permuts,:]

    return data, labels

def transform_labels(labels):
    """ Transforms the label to periodic values.
        Each row of the returning torch tensor is in the form:
            hour, minute, hour cosine, hour sine, minute cosine, minute sine.
    """
    #we will use a dataframe to change the labels since it is more convenient.
    labels_df = pd.DataFrame(labels,columns=['hour', 'minute'])
    labels_df['h_cos'] = np.cos(2 * np.pi * labels_df["hour"] / labels_df["hour"].max())
    labels_df['h_sin'] = np.sin(2 * np.pi * labels_df["hour"] / labels_df["hour"].max())
    labels_df['m_cos'] = np.cos(2 * np.pi * labels_df["minute"] / labels_df["minute"].max())
    labels_df['m_sin'] = np.sin(2 * np.pi * labels_df["minute"] / labels_df["minute"].max())
    return torch.FloatTensor(labels_df.to_numpy())

def train(dataloader, model, loss_fn, optimizer, device, periodic_labels=False):
    """ Applies backpropagation to train the model
    """
    losses = []
    model.train()
    for X, y in dataloader:
        if periodic_labels:
            X, y = X.to(device), y[:, 2:].to(device)
        else:
            X, y = X.to(device), y.to(device)
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

def evaluate(dataloader, model, loss_fn, device, periodic_labels=False):
    """ Used to evaluate the model on unknown data
        during training 
    """
    model.eval()
    losses = []
    with torch.no_grad():
        for X, y in dataloader:
            if periodic_labels:
                X, y = X.to(device), y[:, 2:].to(device)
            else:
                X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            losses.append(loss)
    losses = torch.FloatTensor(losses)
    return losses

def predict(dataloader, model, loss_fn, device, periodic_labels=False):
    """ Returns predictions for the data in the DataLoader 
        as one single batch.
    """
    losses = []
    predictions = []
    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            if periodic_labels:
                X, y = X.to(device), y[:, 2:].to(device)
            else:
                X, y = X.to(device), y.to(device)
            pred = model(X)
            predictions.append(pred.cpu().numpy())
            loss = loss_fn(pred, y)
            losses.append(loss)
    losses = torch.FloatTensor(losses)
    print(f"Avg evaluation loss: {torch.mean(losses):>8f} \n")
    # Normalize values bigger/smaller than the max/min possible
    predictions  = np.vstack(predictions)
    if periodic_labels:
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
    plt.title(plot_name)
    plt.savefig("train_plots/"+plot_name+".png")