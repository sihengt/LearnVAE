import torch
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

data_train = FashionMNIST(
    root="/data",
    train=True,
    transform=ToTensor()
)

data_test = FashionMNIST(
    root="/data",
    train=True,
    transform=ToTensor()
)

LABELS_MAP = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}


train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=64, shuffle=True)

for train_features, train_labels in train_dataloader:
    img = train_features[0].squeeze() # images are already normalized
    label = train_labels[0]
    print(img.size())
    print(f"Label: {label}")
    exit()