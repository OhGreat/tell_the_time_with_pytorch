from cv2 import GaussianBlur
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class ClockDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            t0 = T.RandomApply(torch.nn.ModuleList([
                T.GaussianBlur(kernel_size=(5, 9), sigma=(0.01, 5)),
                #T.RandomErasing(p=0.5, scale=(0.05, 0.2), ratio=(0.2, 3.5)),
                #T.RandomPerspective(0.05, p=0.5),
                ]), p=0.6)
            t1 = T.RandomRotation(degrees=(0,360), expand=True)
            t2 = T.Resize(150)

            composed = T.Compose([t0,t1,t2])
            x = composed(x)

        return x, y
    
    def __len__(self):
        return len(self.data)
