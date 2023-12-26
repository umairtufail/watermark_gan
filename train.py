import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
from discriminator_model import Discriminator
from generator_model import Generator
from vgg_loss import VGGLoss
from dataset import ImageDataset
import torchvision.transforms as transforms
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
])

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)

    dataset = ImageDataset('wm-nowm\\train', transform=transform)
    batch_size = 2  # Adjust based on your needs
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(in_channels=3).to(device)
    discriminator = Discriminator(in_channels=3).to(device)
    vgg_loss = VGGLoss().to(device)

    epochs = 10  # Adjust based on your needs
    lr = 0.0002
    betas = (0.5, 0.999)

    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)

    accumulation_steps = 2
    for epoch in range(epochs):
        for i, (real_images, target_images) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")):
            real_images, target_images = real_images.to(device), target_images.to(device)

            optimizer_G.zero_grad()
            fake_images = generator(real_images)
            fake_images_resized = F.interpolate(fake_images, size=(150, 150), mode='bilinear', align_corners=False)

            target_images_resized = F.interpolate(target_images, size=(150, 150), mode='bilinear', align_corners=False)

            generator_loss = vgg_loss(fake_images_resized, target_images_resized)
            l1_loss = F.l1_loss(fake_images_resized, target_images_resized)

            total_loss = generator_loss + l1_loss

            total_loss.backward()

            if (i + 1) % accumulation_steps == 0:
                optimizer_G.step()
                optimizer_G.zero_grad()

        torch.save(generator.state_dict(), f'generator_epoch_{epoch + 1}.pth')
        torch.save(discriminator.state_dict(), f'discriminator_epoch_{epoch + 1}.pth')
