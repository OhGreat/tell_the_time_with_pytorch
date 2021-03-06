import torch
from torch import nn

class MinutesDistance(nn.Module):
    def __init__(self):
        super(MinutesDistance, self).__init__()

    def forward(self, prediction, target):
        """ distance error in minutes, without truncating integer values
        """
        preds = prediction.T
        labels = target.T
        hour_start_idx = torch.minimum(preds[0], labels[0])
        hour_end_idx = torch.maximum(preds[0], labels[0])
        hour_dist_1 = (hour_end_idx - hour_start_idx)
        hour_dist_2 = hour_start_idx + 12 - hour_end_idx
        hour_dist = torch.minimum(hour_dist_1, hour_dist_2) * 60  # we want the common error to be in minutes

        # minutes common sense error
        mins_start_idx = torch.minimum(preds[1], labels[1])
        mins_end_idx = torch.maximum(preds[1], labels[1])
        mins_dist_1 = (mins_end_idx - mins_start_idx)
        mins_dist_2 = mins_start_idx + 60 - mins_end_idx
        mins_dist = torch.minimum(mins_dist_1, mins_dist_2)

        # return total common sense error in minutes
        return torch.mean(torch.abs(hour_dist + mins_dist))

    def minutes_loss(self, prediction, target):
        """ True distance in minutes with predictions integer truncate.
        """
        preds = torch.trunc(prediction.T)
        labels = target.T
        hour_start_idx = torch.minimum(preds[0], labels[0])
        hour_end_idx = torch.maximum(preds[0], labels[0])
        hour_dist_1 = (hour_end_idx - hour_start_idx)
        hour_dist_2 = hour_start_idx + 12 - hour_end_idx
        hour_dist = torch.minimum(hour_dist_1, hour_dist_2) * 60  # we want the common error to be in minutes

        # minutes common sense error
        mins_start_idx = torch.minimum(preds[1], labels[1])
        mins_end_idx = torch.maximum(preds[1], labels[1])
        mins_dist_1 = (mins_end_idx - mins_start_idx)
        mins_dist_2 = mins_start_idx + 60 - mins_end_idx
        mins_dist = torch.minimum(mins_dist_1, mins_dist_2)

        # return total common sense error in minutes
        return torch.mean(torch.abs(hour_dist + mins_dist))
