import torch

class Encoder(torch.nn.Module):
    def __init__(self, latent_dim):
        # 28 x 28
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,  32, (3,3))
        self.conv2 = torch.nn.Conv2d(32, 64, (3,3))
        self.fc1 = torch.nn.Linear(64 * 24 * 24, 16)
        self.fc2_mean = torch.nn.Linear(16, latent_dim)
        self.fc2_logvar = torch.nn.Linear(16, latent_dim)

    def forward(self, x):
        x = x.unsqueeze(0)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.flatten(x)
        x = torch.relu(self.fc1(x))
        z_mean = self.fc2_mean(x)
        z_logvar = self.fc2_logvar(x)
        return z_mean, z_logvar

