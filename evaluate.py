from classes.ClockDataset import *
from classes.MinutesDistance import *
from classes.Models import *
from utilities import *
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def main():
    # Parameters to tune
    weights_path = 'model_weights/periodic_2'
    approach = "periodic_labels"
    batch_size=64
    data_dir = "data"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_outputs = 4 if approach == "periodic_labels" else 2 

    # define and load model and loss function
    if approach == "baseline":
        model = NN_regression(  input_channels=1,
                            h=150,w=150,
                            n_outputs=n_outputs).to(device)
        model.load_state_dict(torch.load(weights_path))
        loss = nn.MSELoss()
    elif approach == "minute_distance":
        model = NN_regression(  input_channels=1,
                            h=150,w=150,
                            n_outputs=n_outputs).to(device)
        model.load_state_dict(torch.load(weights_path))
        loss = MinutesDistance()
    elif approach == "periodic_labels":
        model = NN_regression_2(  input_channels=1,
                            h=150,w=150,
                            n_outputs=n_outputs).to(device)
        model.load_state_dict(torch.load(weights_path))
        loss = nn.MSELoss()
    else:
        print("Please chose correct approach.")
        exit()

    # load data and create dataset
    data, labels = load_data(data_dir)
    if approach == "periodic_labels":
        labels = transform_labels(labels)
    clock_dataset = ClockDataset(data, labels)
    data_loader = DataLoader(clock_dataset, batch_size=batch_size)

    # make predictions
    cse = MinutesDistance()
    predictions = predict(data_loader, model, loss, device, approach)
    if approach == "periodic_labels":
        # define common sense error
        cse_error = cse(torch.FloatTensor(predictions),torch.FloatTensor(labels[:,:2]))
    else:
        cse_error = cse(torch.FloatTensor(predictions),torch.FloatTensor(labels))
    print(f"Common sense error: {cse_error}")

if __name__ == "__main__":
    main()