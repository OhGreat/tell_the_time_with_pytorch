from torch import nn

class NN_regression(nn.Module):
    """ Convolution model that returns one value as output.
    """
    def __init__(self, input_channels, h, w, n_outputs):
        super(NN_regression, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16)
        )
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=2),
            nn.Dropout(0.3),
            nn.Flatten()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(32768, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, n_outputs) 
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x


class NN_regression_2(nn.Module):
    """ Convolution model that returns one value as output.
    """
    def __init__(self, input_channels, h, w, n_outputs):
        super(NN_regression_2, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=7),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.BatchNorm2d(16)
        )
        self.hidden_layers = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        self.output_layer = nn.Sequential(
            nn.Linear(15488, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, n_outputs)     
        )

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x