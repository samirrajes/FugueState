import torch
import torch.nn as nn
import torch.nn.utils as U

# References:
# https://medium.com/@YasinShafiei/deep-convolution-gan-on-fashion-mnist-using-pytorch-e99619940997
# https://d2l.ai/chapter_generative-adversarial-networks/dcgan.html
# https://arxiv.org/abs/1802.05957
# Lab from Week 7

# Note: ChatGPT was used for sanity checking the data shape and layer configurations within the networks

# DCGAN generator for 128×128 mel‐spectrograms with optional neuron ablation mask in the forward pass
# outputs spectrograms in [0,1]
class AblationGenerator128(nn.Module):
    def __init__(self, latent_dim=100, feature_maps=64, mask=None):
        super().__init__()
        self.latent_dim = latent_dim # size of noise vector z
        self.mask = mask # ablation mask for neurons in the forward pass

        layers = []
        # input z: (B, latent_dim,1,1) to 4x4
        layers += [
            nn.ConvTranspose2d(latent_dim, feature_maps * 8, 4, 1, 0, bias=False), 
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True)
        ]
        # upsample to 8×8
        layers += [
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True)
        ]
        # upsample to 16×16
        layers += [
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True)
        ]
        # upsample to 32×32
        layers += [
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True)
        ]
        # upsample to 64×64
        layers += [
            nn.ConvTranspose2d(feature_maps, feature_maps // 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps // 2),
            nn.ReLU(True)
        ]
        # collect all intermediate layers
        self.net = nn.ModuleList(layers)

        # final upsample to 128×128 and sigmoid to [0,1]
        self.final = nn.ConvTranspose2d(feature_maps // 2, 1, 4, 2, 1, bias=False)
        self.sig   = nn.Sigmoid()

    def forward(self, z):
        h = z
        feats = {}
        for i, layer in enumerate(self.net):
            h = layer(h)
            # apply ablation mask if provided
            if self.mask is not None and i in self.mask:
                h[:, self.mask[i], :, :] = 0.0
            feats[f"layer_{i}"] = h

        out = self.final(h)
        out = self.sig(out) # map into [0,1]
        return out, feats

# Spectral‐normalized discriminator for 128×128 mel‐spectrograms
# Returns raw logits (no sigmoid)
class Discriminator(nn.Module):
    def __init__(self, fmap=64, chans=1):
        super().__init__()
        cfg = [
            (chans, fmap, 4, 2, 1), # 128 to 64
            (fmap, fmap*2, 4, 2, 1), # 64 to 32
            (fmap*2, fmap*4, 4, 2, 1), # 32 to 16
            (fmap*4, fmap*8, 4, 2, 1), # 16 to 8
        ]
        layers = []
        # create layers spectral norm → conv → batch norm → leaky relu
        for inp, out, k, s, p in cfg:
            conv = nn.Conv2d(inp, out, k, s, p, bias=False)
            layers += [
                U.spectral_norm(conv),
                nn.BatchNorm2d(out),
                nn.LeakyReLU(0.2, inplace=False)
            ]
        self.layers = nn.ModuleList(layers)

        # final conv: reduce to single logit (1×1 spatial map)
        final = nn.Conv2d(fmap*8, 1, 4, 1, 0, bias=False)
        self.final = U.spectral_norm(final)

    def forward(self, x):
        feats = []
        h = x
        for i in range(0, len(self.layers), 3):
            h = self.layers[i](h)
            h = self.layers[i+1](h)
            h = self.layers[i+2](h)
            feats.append(h)
        out = self.final(h)
        return out, feats
