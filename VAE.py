import torch
from Decoder import Decoder
from Encoder import Encoder

class VAE(torch.nn.Module):
    def __init__(self, encoder, decoder):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def reparameterize(self, z_mean, z_logvar):
        std = torch.exp(0.5 * z_logvar)
        epsilon = torch.randn_like(std)
        return z_mean + std * epsilon

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x)
        z = self.reparameterize(z_mean, z_logvar)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, z_mean, z_logvar

    def loss(self, x, x_reconstructed, z_mean, z_logvar):
        reconstruction_loss = torch.nn.functional.binary_cross_entropy(x_reconstructed, x, reduction='sum')
        kl_divergence = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        return reconstruction_loss + kl_divergence