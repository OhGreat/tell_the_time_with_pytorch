import numpy as np
import torch
import time
import argparse
from torch import nn
from torch.utils.data import DataLoader, random_split
from classes.ClockDataset import ClockDataset
from classes.CommonSenseError import CommonSenseError
from classes.Models import *
from utilities import *

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-approach', action='store', 
                        dest='approach', type=str,
                        default='periodic_labels')
    parser.add_argument('-data_splits', action='store',
                        dest='data_splits', type=int,
                        nargs='+', default=[16500,1000,500])
    parser.add_argument('-bs', action='store', 
                        dest='batch_size', type=int,
                        default=64)
    parser.add_argument('-lr', action='store', 
                        dest='learning_rate', type=float,
                        default=1e-4)
    parser.add_argument('-epochs', action='store',
                        dest='epochs', type=int,
                        default=100)
    parser.add_argument('-patience', action='store',
                        dest='patience', type=int,
                        default=10)
    parser.add_argument('-weights_name', action='store',
                        dest='weights_name', type=str,
                        default=None)
    parser.add_argument('-save_plots', action='store_true',
                        help="Boolean value. Saves plots when used")
    parser.add_argument('-v', action='store',
                        dest='verbose', type=int,
                        default=1, help="Defines terminal prints intensity. Should be 0,1 or 2.")
    args = parser.parse_args()
    print(args)

    # main approach
    approach = args.approach #periodic_labelscse_loss
    periodic_labels = True if approach == "periodic_labels" else False

    # data parameters
    batch_size = args.batch_size
    input_channels = 1
    img_height = 150
    img_width = 150
    data_splits = args.data_splits  # total must be 180000

    # model parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_outputs = 4 if periodic_labels else 2 
    if approach == "periodic_labels":
        model = NN_regression(  input_channels=input_channels,
                                h=img_height,w=img_width,
                                n_outputs=n_outputs).to(device)
        loss = nn.MSELoss()
    elif approach == "cse_loss":
        model = NN_regression(  input_channels=input_channels,
                                h=img_height,w=img_width,
                                n_outputs=n_outputs).to(device)
        loss = CommonSenseError()
    else:
        print("Please choose a correct mode.")
        exit()
    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = args.epochs
    patience = args.patience
    # extra parameters
    weights_name = args.weights_name
    save_plots = args.save_plots
    verbose = args.verbose

    if verbose > 0:
        print(f"Running pytorch on {device} device.\n")

    # Load data
    data, labels = load_data()
    if verbose > 0:
        print(f"data shape: {data.shape}, labels shape: {labels.shape}")

    # Transform labels to periodic values if needed.
    if periodic_labels:
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
        train_l = train(train_data_loader, model, loss, optimizer, device, approach)
        train_mean = torch.mean(train_l)
        train_losses.append(train_mean)
        print(f"Train loss: {train_mean:>7f}")

        # Evaluation step
        eval_l = evaluate(val_data_loader, model, loss, device, approach)
        eval_mean = torch.mean(eval_l)
        eval_losses.append(eval_mean)
        print(f"Test avg loss: {eval_mean:>8f}")

        # Save new weights if they are better
        if (weights_name != None) and (eval_mean < mean_test_loss):
            print("Found new best weights.")
            mean_test_loss = eval_mean
            torch.save(model.state_dict(), "model_weights/"+weights_name)
            curr_patience = 0
        print()

        # Stop training if patience expired
        if curr_patience >= patience:
            print(f"No new best weights found after {patience} iterations.")
            break
        curr_patience += 1

    end_time = time.time()
    print(f"Training finished in {np.round(end_time-start_time, 3)} seconds.")

    # make predictions on test dataset
    predictions = predict(test_data_loader, model, loss, device, approach)

    cse = CommonSenseError()
    if periodic_labels:
        # transform cosine and sine back to integer values
        true_preds = denormalize_time(predictions)
        # calculate and print common sense error
        cse_error = cse(torch.FloatTensor(true_preds),torch.FloatTensor(targets[:,:2]))
    else:
        cse_error = cse(torch.FloatTensor(predictions),torch.FloatTensor(targets))
    print(f"Common sense error on test dataset: {np.round(cse_error.numpy(),3)}")

    # create and save training plots
    if save_plots:
        train_losses = np.vstack(train_losses)
        test_losses = np.vstack(eval_losses)
        save_train_plot(train_losses, test_losses, weights_name)

if __name__ == "__main__":
    main()