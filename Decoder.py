import torch

class Decoder(torch.nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc1 = torch.nn.Linear(latent_dim, 256)
        self.fc2 = torch.nn.Linear(256, 64 * 7 * 7)
        self.deconv1 = torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(32, 1,  kernel_size=3, stride=2, padding=1, output_padding=1)
    
    def forward(self, z):
        x = torch.relu(self.fc1(z))
        x = torch.relu(self.fc2(x))
        x = x.view(x.size(0), 64, 7, 7)
        x = torch.relu(self.deconv1(x))
        x_reconstructed = torch.sigmoid(self.deconv2(x))
        return x_reconstructed