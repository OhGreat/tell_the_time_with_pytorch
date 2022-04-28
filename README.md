# Tell the time with pytorch

<img src="https://github.com/OhGreat/tell_the_time_NN/blob/main/readme_aux/example_img.png"></img>

This repository is an extract of an assignment from the Computer Science Master's course at Leiden University. It contains a framework used to train and evaluate neural networks on an image dataset of analog clocks. The purpose of this assignment was to tackle the periodic property of time, which would negatively affect the training of our model, if not handled correctly. Details on how this problem was tackled are present in the next section of this document. Pretrained models are also provided to complement the results of our experiments.

## Dataset 

The dataset used for our experiments consists of 18.000 grayscale images of analog clocks in different positions, angles and orientations. The shape of each image is (150,150) and are saved as a numpy arrays in the `data` directory, together with the labels. The labels are also a numpy array of shape (18.000, 2), where for each image we have a target value for hours and minutes. A sample of the dataset can be viewed in the image provided above.

## Problem Formulation

The main problem of trying to predict time with neural networks is that no loss function accounts for the periodicity of time. For example, using a mean squared error loss would confuse the network to think that predictions are further from the ground truth than they actually are. This can be more intuitively illustrated with the following example: if our network predicts 11:59 for an image with target value 12:01, the loss function would penalize strictly the model by returning an error of 11 hours and 58 minutes, where the real distance would be of only 2 minutes.

In order to solve the limitation mentioned above, two main approaches have been used. The first one, employs creating a custom loss function that calculates the error as the minimum distance between the clockwise and counter-clockwise distances of the two clock pointers. The second approach alternatively encodes the labels into periodic values with sine and cosine transforms of the original values. Once the labels have been encoded, a standard error metric like mean absolute error or mean squared error can be used to train our model.

## Prerequisites

To use the repository and run the available scripts, `Python3` ned to be installed together with the following packages:
- `numpy`
- `pytorch` (CUDA is recommended)
- `PIL`
- `matplotlib`

## Usage

Two main python scripts are available in the main directory. `train.py` used for training and `evaluate.py` used to evaluate your trained model weights. Example shell scripts are also available .

The script `train.py` accepts the following arguments:
- `-mode` : defines the problem resolution method. Can be set to *periodic_labels* to use the label tranformation approach, or to *cse_loss* to use the custom common sense error loss.
- `-data_splits` : defines the split sizes for train, test and evaluation set.
- `-bs` : defines the batch size.
- `-lr`: defines the learning rate of the optimizer.
- `-epochs` : defines the maximum number of epochs to train the model.
- `-patience` : defines the number of iterations to wait before stopping the training process, if no new best weights are found.
- `-weights_name` : defines the name of the saved weights.
- `-save_plots` : Boolean value. training plots will be saved when used.
- `-v` : defines the debug prints intensity. Should be 0, 1 or 2.


TODO: describe `evaluate.py`.

## Results and observations

The periodic encoded label configuration worked very well and reaches a common sense error on the test set of around 11 minutes.
The training is also very smooth as we can see from the image below, concluding that this approach is very well suited for the task at hand.

<img src="https://github.com/OhGreat/tell_the_time_NN/tree/main/readme_aux/periodic_labels_losses.png"></img>