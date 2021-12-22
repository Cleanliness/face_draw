from torch.utils.data import Dataset
from torchvision import *
import torchvision.datasets as ds


class FaceLabels(Dataset):
    def __init__(self):
        # load dataset
        self.labels = ds.ImageFolder(root="dataset/faces/input",
                                 transform=transforms.Compose([
                                     transforms.Resize(200),
                                     transforms.CenterCrop(200),
                                     transforms.ToTensor()
                                 ]))
        self.faces = ds.ImageFolder(root="dataset/faces/output",
                                  transform=transforms.Compose([
                                      transforms.Resize(200),
                                      transforms.CenterCrop(200),
                                      transforms.ToTensor()
                                  ]))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        lbl = self.labels[item][0]
        face = self.faces[item][0]

        return {"label": lbl, "face": face}
