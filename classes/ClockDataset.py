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
            ra = T.RandomApply(torch.nn.ModuleList([
                T.GaussianBlur(kernel_size=(5, 9), sigma=(0.01, 5))
            ]), p=0.3)
            re = T.RandomErasing(p=0.5, scale=(0.05, 0.2), ratio=(0.2, 3.5))
            rr = T.RandomRotation(degrees=(0,360), expand=True)
            rp = T.RandomPerspective(0.05, p=0.5)
            rs = T.Resize(150)

            composed = T.Compose([ra,re,rr,rp,rs])
            x = composed(x)

        return x, y
    
    def __len__(self):
        return len(self.data)
