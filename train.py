import numpy as np
import torch
import time
import argparse
from torch import nn
from torch.utils.data import DataLoader, random_split
from classes.ClockDataset import ClockDataset
from classes.MinutesDistance import *
from classes.Models import *
from utilities import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-approach', action='store', 
                        dest='approach', type=str,
                        default='periodic_labels')
    parser.add_argument('-data_dir', action='store', 
                        dest='data_dir', type=str,
                        default='data')
    parser.add_argument('-data_splits', action='store',
                        dest='data_splits', type=int,
                        nargs='+', default=[16500,1000,500])
    parser.add_argument('-data_aug', action='store_true',
                        help="Boolean value. Activates data augmentation \
                            when used")
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
                        default=1, help="Defines terminal prints intensity. \
                            Should be 0,1 or 2.")
    args = parser.parse_args()
    if args.verbose > 0:
        print()
        print("Arguments:")
        print(args)

    # main approach
    approach = args.approach #periodic_labelscse_loss
    periodic_labels = True if approach == "periodic_labels" else False

    # data parameters
    data_dir = args.data_dir
    batch_size = args.batch_size
    input_channels = 1
    img_height = 150
    img_width = 150
    data_splits = args.data_splits  # total must be 180000

    # model parameters
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_outputs = 4 if periodic_labels else 2
    model = NN_regression_2(  input_channels=input_channels,
                                h=img_height,w=img_width,
                                n_outputs=n_outputs).to(device)
    if approach == "periodic_labels" or approach == "baseline":
        loss = nn.MSELoss()
    elif approach == "minute_distance":
        loss = MinutesDistance()
    else:
        print("Please choose a correct mode.")
        exit()
    learning_rate = args.learning_rate
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    epochs = args.epochs
    patience = args.patience

    # extra control parameters
    weights_name = args.weights_name
    save_plots = args.save_plots
    verbose = args.verbose

    if verbose > 0:
        print(f"Running pytorch on {device} device.\n")

    # Load data
    data, labels = load_data(data_dir)
    # split test set as to not apply data augmentation
    test_data, test_labels = data[:data_splits[2],:], labels[:data_splits[2],:]
    data, labels = data[data_splits[2]:,:], labels[data_splits[2]:,:]
    # Transform labels to periodic values if required.
    if periodic_labels:
        labels = transform_labels(labels)
        test_labels = transform_labels(test_labels)
    if verbose > 0:
        print(f"train data shape: {data.shape}, train labels shape: {labels.shape}")
        print(f"test data shape: {test_data.shape}, test labels shape: {test_labels.shape}")

    # Create main dataset
    clock_train_dataset = ClockDataset(data, labels, transform=args.data_aug)
    clock_test_datset = ClockDataset(test_data, test_labels, transform=False)
    # Split dataset into train and validation sets
    train_data, val_data = random_split(clock_train_dataset, data_splits[:2])
    # Create data loaders
    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(clock_test_datset, batch_size=batch_size)
    if verbose > 0:
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
        print(f"Training loss: {train_mean:>7f}")

        # Evaluation step
        eval_l = evaluate(val_data_loader, model, loss, device, approach)
        eval_mean = torch.mean(eval_l)
        eval_losses.append(eval_mean)
        print(f"Evaluation loss: {eval_mean:>8f}")

        # Save new weights if they are better
        if eval_mean < mean_test_loss:
            curr_patience = 0
            if weights_name != None: 
                print("Saving new best weights.")
                mean_test_loss = eval_mean
                torch.save(model.state_dict(), "model_weights/"+weights_name)
            curr_patience = 0
        print()

        # Stop training if patience expired
        if curr_patience >= patience:
            print(f"No new best weights found after {patience} iterations.")
            break
        curr_patience += 1  # increment patience if new best not found

    end_time = time.time()
    print(f"Training finished in {np.round(end_time-start_time, 3)} seconds.")

    # make predictions on test dataset
    predictions = predict(test_data_loader, model, loss, device, approach)

    # calculate and print common sense error
    cse = MinutesDistance()
    if periodic_labels:
        # transform cosine and sine back to integer values
        cse_error = cse.minutes_loss(torch.FloatTensor(predictions),
                                    torch.FloatTensor(test_labels[:,:2]))
    else:
        cse_error = cse.minutes_loss(torch.FloatTensor(predictions),
                                    torch.FloatTensor(test_labels))
    print(f"Mean minute distance loss for the test dataset: {np.round(cse_error.numpy(),3)}")

    # create and save training plots
    if save_plots:
        if weights_name == None:  # make sure we have a name for thet plots
            weights_name = approach
        train_losses = np.vstack(train_losses)
        test_losses = np.vstack(eval_losses)
        save_train_plot(train_losses, test_losses, weights_name)

if __name__ == "__main__":
    main()