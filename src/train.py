# src/train.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.models import AblationGenerator128, Discriminator
from src.utils import mel_loss

# References:
# https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html

# function to init weights for the models
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

# define training loop for GAN
def train_gan(loader, device, output_dir, cfg):
    # instantiate models
    G = AblationGenerator128(latent_dim=cfg['latent_dim'], feature_maps=cfg['feature_maps'], mask=None).to(device)
    D = Discriminator().to(device)

    # weight initialization
    G.apply(weights_init)
    D.apply(weights_init)

    # optimizers and loss
    criterion = nn.BCEWithLogitsLoss()
    optD = optim.Adam(D.parameters(), lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']))
    optG = optim.Adam(G.parameters(), lr=cfg['lr'], betas=(cfg['beta1'], cfg['beta2']))

    for epoch in range(1, cfg['epochs'] + 1):
        G.train(); D.train()
        loop = tqdm(loader, desc=f"Epoch {epoch}/{cfg['epochs']}")

        for specs in loop:
            specs = specs.to(device)
            real_spec = F.interpolate(specs, size=(cfg['n_mels'], 128), mode='bilinear', align_corners=False)
            real_in = (real_spec + 1) / 2  # [0,1]
            B = real_in.size(0)

            # D step
            optD.zero_grad()
            out_real, _ = D(real_in)
            out_real = out_real.view(B, -1).mean(1)
            label_real = torch.ones(B, device=device)
            loss_real = criterion(out_real, label_real)

            z = torch.randn(B, cfg['latent_dim'], device=device).view(B, cfg['latent_dim'], 1, 1)
            fake_spec, _ = G(z)
            fake_in = fake_spec

            out_fake, _ = D(fake_in.detach())
            out_fake = out_fake.view(B, -1).mean(1)
            label_fake = torch.zeros(B, device=device)
            loss_fake = criterion(out_fake, label_fake)

            lossD = 0.5 * (loss_real + loss_fake)
            lossD.backward()
            optD.step()

            # G step
            optG.zero_grad()
            out_gen, _ = D(fake_in)
            out_gen = out_gen.view(B, -1).mean(1)
            adv_loss = criterion(out_gen, torch.ones(B, device=device))

            mel = mel_loss(real_spec, fake_spec)
            lossG = adv_loss + 0.1 * mel

            lossG.backward()
            optG.step()

            loop.set_postfix({"Loss_D": lossD.item(), "Loss_G": lossG.item()})

        # end of epoch sampling
        G.eval()
        with torch.no_grad():
            sample_z = torch.randn(5, cfg['latent_dim'], device=device).view(5, cfg['latent_dim'], 1, 1)
            samples, _ = G(sample_z)

        fig, axes = plt.subplots(1, 5, figsize=(15, 3))
        for i, ax in enumerate(axes):
            ax.imshow(samples[i, 0].cpu(), origin='lower', aspect='auto')
            ax.axis('off')
            ax.set_title(f"Sample {i+1}")
        plt.suptitle(f"Epoch {epoch} Generated Samples")
        plt.tight_layout()
        plt.show()

        G.train()

        # checkpoint
        if epoch % cfg['checkpoint_epoch'] == 0:
            torch.save(G.state_dict(), os.path.join(output_dir, f"G_ep{epoch}.pth"))
            torch.save(D.state_dict(), os.path.join(output_dir, f"D_ep{epoch}.pth"))

    return G
