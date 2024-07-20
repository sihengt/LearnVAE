import torch
import matplotlib.pyplot as plt
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor
from VAE import VAE
from Encoder import Encoder
from Decoder import Decoder

LATENT_DIM = 20
N_EPOCHS = 10

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataloader = torch.utils.data.DataLoader(data_train, batch_size=64, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(data_test, batch_size=64, shuffle=True)

# for train_features, train_labels in train_dataloader:
#     img = train_features[0].squeeze() # images are already normalized
#     label = train_labels[0]
#     print(img.size())
#     print(f"Label: {label}")

encoder = Encoder(LATENT_DIM).to(device)
decoder = Decoder(LATENT_DIM).to(device)
model = VAE(encoder, decoder).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-2)

def visualize_and_save_reconstruction(epoch):
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        data, _ = next(iter(train_dataloader))
        data = data.to(device)  # Move data to the correct device
        data = data.to(torch.float32)
        x_reconstructed, _, _ = model(data)

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(data[0].cpu().view(28, 28), cmap='gray')  # Move data back to CPU for visualization
        axes[0].set_title('Original')
        axes[1].imshow(x_reconstructed[0].cpu().view(28, 28), cmap='gray')  # Move reconstructed data back to CPU
        axes[1].set_title('Reconstructed')
        plt.savefig(f'reconstructed_epoch_{epoch}.png')
        plt.close(fig)

for epoch in range(1, N_EPOCHS + 1):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_dataloader):
        data = data.to(device)
        data = data.to(torch.float32)
        optim.zero_grad()
        x_reconstructed, z_mean, z_logvar = model(data)
        loss = model.loss(data, x_reconstructed, z_mean, z_logvar)
        loss.backward()
        train_loss += loss.item()
        optim.step()
    print(f'====> Epoch: {epoch} Average loss: {train_loss / len(train_dataloader.dataset):.4f}')
    visualize_and_save_reconstruction(epoch)

