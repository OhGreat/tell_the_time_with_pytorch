# Tell the time with pytorch

<img src="https://github.com/OhGreat/tell_the_time_NN/blob/main/readme_aux/example_img.png"></img>

This repository is an extract of an assignment from the Computer Science Master's course at Leiden University. It contains a framework used to train and evaluate neural networks on an image dataset of analog clocks. The purpose of this assignment was to tackle the cyrcular property of time, which would negatively affect the training of our model if not approached correctly. Details on how this problem was tackled are present in the next section of this document. Pretrained models are also provided to complement the results of our experiments.

## Dataset 

The dataset used for our experiments cosists of 18.000 grayscale images of analog clocks in different positions, angles and orientations. The shape of each image is (150,150) and are saved as a numpy array in the `data` directory together with all the labels. The labels on the other hand, are a numpy array of shape (18.000, 2), where for each image we have a value for hours and minutes.

## Problem Formulation

The main problem of trying to predict time with neural networks is that no loss accounts for the periodicity of time. For example, using a mean squared error loss would confuse the network to think that predictions are further from the ground truth, than they actually are. This can be more intuitively explained with the following example: if our network predicts 11:59 for an image with target value 12:01, the loss function would penalize strictly the model, since it would return an error of 11 hour and 58 minutes, where the real distance would be of only 2 minutes.

In order to solve this issue two main approaches have been used. In the first one, we create a custom loss function. The loss function calculates the error as the minimum between the  
In the second approach, the labels are encoded into periodic values with sine and cosine transforms.

## Usage

## Results and observations
