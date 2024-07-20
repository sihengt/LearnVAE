import torch

class Encoder(torch.nn.Module):
    def __init__(self, latent_dim):
        # 28 x 28
        super(Encoder, self).__init__()
        self.conv1 = torch.nn.Conv2d(1,  32, kernel_size=3, stride=2, padding=1)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.fc1 = torch.nn.Linear(64 * 7 * 7, 256)
        self.fc2_mean = torch.nn.Linear(256, latent_dim)
        self.fc2_logvar = torch.nn.Linear(256, latent_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten the tensor for the fully connected layer
        x = torch.relu(self.fc1(x))
        z_mean = self.fc2_mean(x)
        z_logvar = self.fc2_logvar(x)
        return z_mean, z_logvar

