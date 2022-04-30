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
                T.GaussianBlur(kernel_size=(5, 9), sigma=(0.2, 2))
            ]), p=0.3)
            rp = T.RandomPerspective(0.3, p=0.3)
            rr = T.RandomRotation(degrees=(0,360), expand=True)
            re = T.RandomErasing(p=0.6, scale=(0.05, 0.25), ratio=(0.2, 3.5))
            rs = T.Resize(150)

            composed = T.Compose([ra,rp,rr,re,rs])
            x = composed(x)

        return x, y
    
    def __len__(self):
        return len(self.data)
