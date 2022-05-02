from classes.ClockDataset import *
from classes.MinutesDistance import *
from classes.Models import *
from utilities import *
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def main():
    # Parameters to tune
    weights_path = 'model_weights/periodic'
    approach = "periodic_labels"
    batch_size=64
    data_dir = "data"

    # load model and weights
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_outputs = 4 if approach == "periodic_labels" else 2 
    model = NN_regression(  input_channels=1,
                            h=150,w=150,
                            n_outputs=n_outputs).to(device)
    model.load_state_dict(torch.load(weights_path))

    # define loss function
    if approach == "periodic_labels" or approach == "baseline":
        loss = nn.MSELoss()
    else:
        loss = MinutesDistance()

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
        true_preds = denormalize_time(predictions)
        # define common sense error
        cse_error = cse(torch.FloatTensor(true_preds),torch.FloatTensor(labels[:,:2]))
    else:
        cse_error = cse(torch.FloatTensor(predictions),torch.FloatTensor(labels))
    print(f"Common sense error: {cse_error}")

if __name__ == "__main__":
    main()