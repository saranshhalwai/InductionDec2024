import pdb
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid

transform = transforms.Compose([transforms.Resize(255),
                                 transforms.CenterCrop(224),
                                 transforms.ToTensor()])

dataset = datasets.ImageFolder('path/to/data', transform=transform)

def show(tensor, ch=1, size=(512, 512), num=25):
    data = tensor.detach().cpu().view(-1, ch, *size)
    grid = make_grid(data[:num], nrow=5).permute(1, 2, 0)
    plt.imshow(grid)
    plt.show()

epoch = 1000
cur_iter = 0
info_iter = 300
mean_gen_loss = 0
mean_disc_loss = 0

z_dim = 64
lr = 0.0001
loss = nn.BCEWithLogitsLoss()
batch_size = 1024
device = "cuda"

dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

def genBlock(inp_nodes, out_nodes):
    return nn.Sequential(
        nn.Linear(inp_nodes, out_nodes),
        nn.BatchNorm1d(out_nodes),
        nn.ReLU()
    )

def gen_noise(batch_size, z_dim):
    return torch.randn(batch_size, z_dim).to(device)

class Generator(nn.Module):
    def __init__(self, z_dim=64, o_dim=262144, h_dim=120):
        super().__init__()
        self.z_dim = z_dim
        self.o_dim = o_dim
        self.h_dim = h_dim
        self.gen = nn.Sequential(
            genBlock(z_dim, h_dim),
            genBlock(h_dim, h_dim * 2),
            genBlock(h_dim * 2, h_dim * 4),
            genBlock(h_dim * 4, h_dim * 8),
            genBlock(h_dim * 8, o_dim),
            nn.Sigmoid(),
        )

    def forward(self, noise):
        return self.gen(noise)

def discBlock(inp_nodes, out_nodes):
    return nn.Sequential(
        nn.Linear(inp_nodes, out_nodes),
        nn.LeakyReLU(0.2)
    )

class Discriminator(nn.Module):
    def __init__(self, inp_dim=262144, hidden_dim=256):
        super().__init__()
        self.inp_dim = inp_dim
        self.hidden_dim = hidden_dim
        self.disc = nn.Sequential(
            discBlock(inp_dim, hidden_dim * 4),
            discBlock(hidden_dim * 4, hidden_dim * 2),
            discBlock(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, image):
        return self.disc(image)

# Optimizers
gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)

disc = Discriminator().to(device)
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)

def gen_loss(loss_func, gen, disc, batch_size, z_dim):
    noise = gen_noise(batch_size, z_dim)
    fake = gen(noise)
    pred = disc(fake)
    target = torch.ones_like(pred)
    return loss_func(pred, target)

def disc_loss(loss_func, gen, disc, batch_size, z_dim, real):
    noise = gen_noise(batch_size, z_dim)
    fake = gen(noise)
    disc_fake = disc(fake.detach())
    disc_fake_target = torch.zeros_like(disc_fake)
    disc_fake_loss = loss_func(disc_fake, disc_fake_target)

    disc_real = disc(real)
    disc_real_target = torch.ones_like(disc_real)
    disc_real_loss = loss_func(disc_real, disc_real_target)

    return (disc_fake_loss + disc_real_loss) / 2

for epoch in range(epoch):
    mean_disc_loss_list = []
    mean_gen_loss_list = []
    iters_list = []

    for real_image, _ in tqdm(dataloader):
        disc_opt.zero_grad()
        cur_batch_size = len(real_image)
        real_image = real_image.view(cur_batch_size, -1).to(device)
        disc_losses = disc_loss(loss, gen, disc, cur_batch_size, z_dim, real_image)
        disc_losses.backward()
        disc_opt.step()

        gen_opt.zero_grad()
        gen_losses = gen_loss(loss, gen, disc, cur_batch_size, z_dim)
        gen_losses.backward()
        gen_opt.step()

        mean_disc_loss += disc_losses.item() / info_iter
        mean_gen_loss += gen_losses.item() / info_iter
        mean_disc_loss_list.append(mean_disc_loss)
        mean_gen_loss_list.append(mean_gen_loss)

        if cur_iter % info_iter == 0 and cur_iter > 0:
            fake_noise = gen_noise(cur_batch_size, z_dim)
            fake = gen(fake_noise)
            show(real_image)
            show(fake)
            print(f"{epoch} : step {cur_iter}, Generator loss : {mean_gen_loss}, Discriminator Loss : {mean_disc_loss}")
            mean_gen_loss, mean_disc_loss = 0, 0

        iters_list.append(cur_iter)
        cur_iter += 1
