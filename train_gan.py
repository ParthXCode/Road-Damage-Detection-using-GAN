import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ---------------- CONFIG ----------------
LATENT_DIM = 100
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 200
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- DATA ----------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.15, contrast=0.15),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder("../pothole_crops", transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------- GENERATOR ----------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(LATENT_DIM, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)


# ---------------- DISCRIMINATOR ----------------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).view(-1, 1)


# ---------------- INIT ----------------
G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

criterion = nn.BCELoss()
g_opt = torch.optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
d_opt = torch.optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

# ---------------- TRAIN ----------------
for epoch in range(EPOCHS):
    for imgs, _ in loader:
        imgs = imgs.to(DEVICE)

        real = torch.ones(imgs.size(0), 1).to(DEVICE)
        fake = torch.zeros(imgs.size(0), 1).to(DEVICE)

        z = torch.randn(imgs.size(0), LATENT_DIM, 1, 1).to(DEVICE)
        gen_imgs = G(z)

        # Train D
        d_loss = criterion(D(imgs), real) + criterion(D(gen_imgs.detach()), fake)
        d_opt.zero_grad()
        d_loss.backward()
        d_opt.step()

        # Train G
        g_loss = criterion(D(gen_imgs), real)
        g_opt.zero_grad()
        g_loss.backward()
        g_opt.step()

    print(f"Epoch {epoch+1}/{EPOCHS} | D: {d_loss.item():.4f} | G: {g_loss.item():.4f}")

# ---------------- SAVE ----------------
os.makedirs("../models", exist_ok=True)
torch.save(G.state_dict(), "../models/generator.pth")

print("Improved GAN training complete.")