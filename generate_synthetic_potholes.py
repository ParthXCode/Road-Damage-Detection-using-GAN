import os
import torch
from torchvision.utils import save_image
from train_gan import Generator, LATENT_DIM, DEVICE

os.makedirs("../synthetic_potholes", exist_ok=True)

G = Generator().to(DEVICE)
G.load_state_dict(torch.load("../models/generator.pth"))
G.eval()

for i in range(500):
    z = torch.randn(1, LATENT_DIM, 1, 1).to(DEVICE)
    img = G(z)
    save_image(img, f"../synthetic_potholes/pothole_{i}.png", normalize=True)

print("Synthetic potholes generated.")