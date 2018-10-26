import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, params):
        super(Generator, self).__init__()
        self.z_dim = params['z_dim']
        self.slope = params['slope']
        self.num_channels = params['num_channels']

        self.deconv = nn.Sequential(
            # dim: z_dim x 1 x 1
            nn.ConvTranspose2d(self.z_dim, 256, 4, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.slope, True),
            # dim:   256 x 4 x 4
            nn.ConvTranspose2d(256, 128, 4, 2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.slope, True),
            # dim: 128 x 10 x 10
            nn.ConvTranspose2d(128, 64, 4, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.slope, True),
            # dim: 64 x 13 x 13
            nn.ConvTranspose2d(64, 32, 4, 2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.slope, True),
            # dim: 32 x 28 x 28
            nn.Conv2d(32, self.num_channels, 1, 1),
            # dim: out_channels x 28 x 28
            nn.Tanh()
        )

        self.apply(weights_init)

    def forward(self, x):
        x = self.deconv(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, params):
        super(Discriminator, self).__init__()
        self.z_dim = params['z_dim']
        self.slope = params['slope']
        self.num_channels = params['num_channels']
        self.dropout = params['dropout']

        self.inference_x = nn.Sequential(
            # dim: num_channels 28 x 28
            nn.Conv2d(self.num_channels, 64, 4, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.slope, True),
            nn.Dropout2d(self.dropout),
            # dim: 64 x 13 x 13
            nn.Conv2d(64, 128, 4, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.slope, True),
            nn.Dropout2d(self.dropout),
            # dim: 128 x 10 x 10
            nn.Conv2d(128, 256, 4, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.slope, True),
            nn.Dropout2d(self.dropout),
            # dim: 256 x 4 x 4
            nn.Conv2d(256, 512, 4, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.slope, True),
            nn.Dropout2d(self.dropout)
            # output dim: 512 x 1 x 1
        )

        self.inference_z = nn.Sequential(
            nn.Conv2d(self.z_dim, 512, 1, 1, bias=False),
            nn.LeakyReLU(self.slope, True),
            # nn.Dropout2d(self.dropout),

            nn.Conv2d(512, 512, 1, 1, bias=False),
            nn.LeakyReLU(self.slope, True),
            # nn.Dropout2d(self.dropout)
        )

        self.inference_joint = nn.Sequential(
            torch.nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            torch.nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            torch.nn.Linear(512, 1),
            torch.nn.Sigmoid()
        )

        self.apply(weights_init)

    def forward(self, x, z):
        x = self.inference_x(x)
        z = self.inference_z(z)
        out = self.inference_joint(torch.cat((x, z), 1).squeeze())
        return out


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()
        self.z_dim = params['z_dim']
        self.slope = params['slope']
        self.num_channels = params['num_channels']

        self.inference = nn.Sequential(
            # dim: num_channels x 32 x 32
            nn.Conv2d(self.num_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(self.slope, True),
            # dim: 32 x 28 x 28
            nn.Conv2d(32, 64, 4, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(self.slope, True),
            # dim: 64 x 13 x 13
            nn.Conv2d(64, 128, 4, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(self.slope, True),
            # dim: 128 x 10 x 10
            nn.Conv2d(128, 256, 4, 2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(self.slope, True),
            # dim: 256 x 4 x 4
            nn.Conv2d(256, 512, 4, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.slope, True),
            # dim: 512 x 1 x 1
            nn.Conv2d(512, 512, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(self.slope, True),
            # dim: 512 x 1 x 1
            nn.Conv2d(512, self.z_dim, 1, 1)
            # output dim: z_dim x 1 x 1
        )

        self.apply(weights_init)

    def forward(self, x):
        x = self.inference(x)
        return x