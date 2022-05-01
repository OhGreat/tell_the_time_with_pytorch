# Tell the time with pytorch


This repository is an extract of an assignment from the Computer Science Master's course at Leiden University. It contains a framework used to train and evaluate neural networks on an image dataset of analog clocks. The purpose of this assignment was to tackle the periodic property of time, which would negatively affect the training of our model, if not handled correctly. Details on how this problem was tackled are present in the next section of this document. Pretrained models are also provided to complement the results of our experiments.

## Dataset 

<img src="https://github.com/OhGreat/tell_the_time_NN/blob/main/readme_aux/example_img.png"></img>

The dataset used for our experiments consists of 18.000 grayscale images of analog clocks in different positions, angles and orientations. The shape of each image is (150,150) and are saved as a numpy arrays in the `data` directory, together with the labels. The labels are also a numpy array of shape (18.000, 2), where for each image we have a target value for hours and minutes. A sample of the dataset can be viewed in the image provided above.

## Problem Formulation

The main problem of trying to predict time with neural networks is that no loss function accounts for the periodicity of time. For example, using a mean squared error loss would confuse the network to think that predictions are further from the ground truth than they actually are. This can be more intuitively illustrated with the following example: if our network predicts 11:59 for an image with target value 12:01, the loss function would penalize strictly the model by returning an error of 11 hours and 58 minutes, where the real distance would be of only 2 minutes.

In order to solve the limitation mentioned above, two main approaches have been used. The first one, employs creating a custom loss function that calculates the error as the minimum distance between the clockwise and counter-clockwise distances of the two clock pointers. The second approach alternatively encodes the labels into periodic values with sine and cosine transforms of the original values. Once the labels have been encoded, a standard error metric like mean absolute error or mean squared error can be used to train our model.

## Prerequisites

To use the repository and run the available scripts, `Python3` needs to be installed together with the following packages:
- `numpy`
- `pytorch` (CUDA is recommended)
- `PIL`
- `matplotlib`

## Usage

Two main python scripts are available in the main directory. `train.py` that is used for training and `evaluate.py` which is used to evaluate trained models. Shell scripts for setting parameters on the main python scripts can be found in the `example_scripts` directory. In addition, jupyter notebooks for training and evaluationg models are available in the `notebooks` directory. 

The script `train.py` accepts the following arguments:
- `-approach` : defines the problem resolution method. Can be set to *"baseline"* to run the basic experiment, to *"periodic_labels"* to use the label tranformation approach,  to *"minute_distance"* to use the custom minute distance loss.
- `-data_splits` : defines the split sizes for train, test and evaluation set.
- `-bs` : defines the batch size.
- `-lr` : defines the learning rate of the optimizer used for training.
- `-epochs` : defines the maximum number of epochs to train the model.
- `-patience` : defines the number of iterations to wait before stopping the training process, if no new best weights are found.
- `-weights_name` : defines the name of the saved weights.
- `-save_plots` : Boolean value. training plots will be saved when used.
- `-v` : defines the debug prints intensity. Should be 0, 1 or 2.

To train an example network with the periodic-labelled approach, run the following python script from the `main directory`:
```
python train.py -approach "periodic_labels" -weights_name "example_periodic" -v 1
```

To use the custom minute-loss approach instead, run the following python script from the `main directory`:
```
python train.py -approach "cse_loss" -lr 1e-5 -weights_name "example_cse" -v 1
```

To run the bash scripts execute the following commands from the `main directory`, by substituting <bash_script.sh> with the name of the script you want to use in the `example_scripts` directory:
```
chmod +x example_scripts/<bash_script.sh>
./example_scripts/<bash_script.sh>
``` 

The main argumnts required to use the `evaluate.py` are the *weights_path* and the *approach*. These can be set directly inside the script itself between lines 12-15, together with the batch size and data path. To run the python script after setting the arguments, run from the main directory the following command:
```
python evaluate.py
```

## Results and observations

All configurations have been tested with very similar parameters in order to make as fair of a comparison as possible:
- `data splits`: 16500 samples for training set, 1000 for the evaluation set and 500 for the test set.
- `neural network`: the same CNN configuration was used for all approaches
- `learning rate`: 1e-4 for the periodic labels approach and 1e-5 for the baseline and minutes-loss approaches.
- `batch size`: 64
- `maximum epochs`: 200
- `patience`: 10

In order to compare all the approaches, the minutes-distance loss was calculated after training on the test set and is reported below.

The baseline configuration simply uses the mean squared error as a metric without any label transformation and is used to bechmark our approaches.
This approach reaches a minutes-distance loss of around 85.6 minutes on the test set and although the validation loss does not descend very smoothly, it  converges after around 40-60 epochs.

The minutes-distance loss approach, on the other hand, achieves a mean loss of 48,8 minutes, which although is better than the baseline approach, the results are still unsatisfactory and the training very unstable.

Finally, the periodic encoded label configuration outperforms the other two and reaches a mean minutes-distance loss on the test set of around 12 minutes.
The training was also very smooth, concluding that this approach is very well suited for the task at hand.

<img src="https://github.com/OhGreat/tell_the_time_NN/blob/main/readme_aux/periodic_labels_losses.png"></img>

## TODO

- ~~fix custom loss approach~~
- fix custom loss notebook
- add args parser in evaluate script
- add evaluate.py usage description
