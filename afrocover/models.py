"""
StyleGAN2 Models for AfroCover

Implementation of StyleGAN2 Generator and Discriminator for album cover generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MappingNetwork(nn.Module):
    """StyleGAN2 Mapping Network (Z -> W)"""
    
    def __init__(self, z_dim=512, w_dim=512, num_layers=8):
        super().__init__()
        
        layers = []
        for i in range(num_layers):
            in_dim = z_dim if i == 0 else w_dim
            layers.extend([
                nn.Linear(in_dim, w_dim),
                nn.LeakyReLU(0.2)
            ])
        
        self.mapping = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.mapping(z)


class ModulatedConv2d(nn.Module):
    """Modulated Convolution Layer"""
    
    def __init__(self, in_channels, out_channels, kernel_size, style_dim, upsample=False):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.upsample = upsample
        
        # Convolution weight
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        
        # Style modulation
        self.style_proj = nn.Linear(style_dim, in_channels)
        
        # Noise injection
        self.noise_strength = nn.Parameter(torch.zeros(1))
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(out_channels))
        
    def forward(self, x, style, noise=None):
        batch_size = x.shape[0]
        _, _, height, width = x.shape

        # Project style to per-channel modulation (B, in_channels)
        style = self.style_proj(style)  # [B, in_channels]

        # Prepare modulation: reshape to [B,1,in_ch,1,1] for broadcasting
        style = style.view(batch_size, 1, self.in_channels, 1, 1)

        # Expand conv weight to include batch dimension: [1, out_ch, in_ch, k, k]
        weight = self.weight.unsqueeze(0)

        # Modulate
        weight = weight * style  # [B, out_ch, in_ch, k, k]

        # Demodulation
        demod = torch.rsqrt((weight ** 2).sum(dim=[2, 3, 4], keepdim=True) + 1e-8)
        weight = weight * demod

        # Reshape for grouped convolution
        weight = weight.view(batch_size * self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        x = x.view(1, batch_size * self.in_channels, height, width)

        # Upsample if needed (apply to feature map before grouped conv)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

        # Convolution using grouped conv
        x = F.conv2d(x, weight, groups=batch_size, padding=self.kernel_size//2)
        x = x.view(batch_size, self.out_channels, x.shape[2], x.shape[3])

        # Prepare and add noise
        out_h, out_w = x.shape[2], x.shape[3]

        if noise is None:
            # Create single-channel noise per-pixel, which will broadcast over channels
            noise = torch.randn(batch_size, 1, out_h, out_w, device=x.device)
        else:
            # If provided, ensure spatial dims match output; if not, resize
            if noise.dim() == 4 and (noise.shape[2] != out_h or noise.shape[3] != out_w):
                noise = F.interpolate(noise, size=(out_h, out_w), mode='bilinear', align_corners=False)
            # Reduce to single channel (StyleGAN uses single-channel noise scaled per output channel)
            if noise.dim() == 4 and noise.shape[1] != 1:
                noise = noise.mean(dim=1, keepdim=True)

        x = x + noise * self.noise_strength

        # Add bias
        x = x + self.bias.view(1, -1, 1, 1)

        return x


class StyleGAN2Generator(nn.Module):
    """StyleGAN2 Generator"""
    
    def __init__(
        self,
        z_dim=512,
        w_dim=512,
        img_resolution=256,
        img_channels=3,
        channel_multiplier=1.0,
    ):
        super().__init__()
        
        self.z_dim = z_dim
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_layers = int(math.log2(img_resolution)) - 1
        self.channel_multiplier = channel_multiplier
        
        # Mapping network
        self.mapping = MappingNetwork(z_dim, w_dim)
        
        # Helper for scaled channels
        def _scaled_channels(value, minimum=16, maximum=512):
            scaled = int(value * self.channel_multiplier)
            return max(minimum, min(maximum, scaled))

        const_channels = _scaled_channels(512, minimum=32)

        # Synthesis network
        self.const = nn.Parameter(torch.randn(1, const_channels, 4, 4))
        
        # Progressive layers
        self.layers = nn.ModuleList()
        in_channels = const_channels
        
        for i in range(self.num_layers):
            raw_channels = min(512, 512 // (2 ** max(0, i - 4)))
            out_channels = _scaled_channels(raw_channels)
            
            self.layers.append(nn.ModuleList([
                ModulatedConv2d(in_channels, out_channels, 3, w_dim, upsample=i > 0),
                nn.LeakyReLU(0.2),
                ModulatedConv2d(out_channels, out_channels, 3, w_dim),
                nn.LeakyReLU(0.2)
            ]))
            
            in_channels = out_channels
        
        # Final RGB output
        self.to_rgb = ModulatedConv2d(in_channels, img_channels, 1, w_dim)
        
    def forward(self, z, truncation_psi=1.0):
        batch_size = z.shape[0]
        
        # Map to W space
        w = self.mapping(z)
        
        # Start with constant
        x = self.const.repeat(batch_size, 1, 1, 1)
        
        # Progressive synthesis
        for layer_group in self.layers:
            upconv, act1, conv, act2 = layer_group
            
            # Add noise
            noise = torch.randn_like(x) if self.training else torch.zeros_like(x)
            
            x = upconv(x, w, noise)
            x = act1(x)
            
            noise = torch.randn_like(x) if self.training else torch.zeros_like(x)
            x = conv(x, w, noise)
            x = act2(x)
        
        # Convert to RGB
        noise = torch.randn_like(x[:, :1]) if self.training else torch.zeros_like(x[:, :1])
        rgb = self.to_rgb(x, w, noise)
        rgb = torch.tanh(rgb)
        
        return rgb


class StyleGAN2Discriminator(nn.Module):
    """StyleGAN2 Discriminator"""
    
    def __init__(self, img_resolution=256, img_channels=3, channel_multiplier=1.0):
        super().__init__()
        
        self.img_resolution = img_resolution
        self.img_channels = img_channels
        self.num_layers = int(math.log2(img_resolution)) - 1
        self.channel_multiplier = channel_multiplier
        
        # Progressive layers
        self.layers = nn.ModuleList()
        
        # From RGB
        base_channels = max(32, min(512, int(32 * self.channel_multiplier)))
        self.from_rgb = nn.Conv2d(img_channels, base_channels, 1)
        
        # Downsampling layers
        in_channels = base_channels
        for i in range(self.num_layers):
            raw_channels = min(512, 32 * (2 ** (i + 1)))
            out_channels = max(base_channels, int(raw_channels * self.channel_multiplier))
            
            self.layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.LeakyReLU(0.2),
                nn.AvgPool2d(2)
            ))
            
            in_channels = out_channels
        
        # Final layers
        self.final_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.final_linear = nn.Linear(in_channels, 1)
        
    def forward(self, x):
        x = self.from_rgb(x)
        
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_conv(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.shape[0], -1)
        x = self.final_linear(x)
        
        return x


def test_models():
    """Test model creation and forward pass"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test Generator
    print("Testing StyleGAN2 Generator...")
    generator = StyleGAN2Generator(
        z_dim=512,
        w_dim=512,
        img_resolution=256,
        img_channels=3,
        channel_multiplier=0.5,
    ).to(device)
    
    z = torch.randn(2, 512).to(device)
    fake_images = generator(z)
    print(f"Generator output shape: {fake_images.shape}")
    
    # Test Discriminator
    print("Testing StyleGAN2 Discriminator...")
    discriminator = StyleGAN2Discriminator(
        img_resolution=256,
        img_channels=3,
        channel_multiplier=0.5,
    ).to(device)
    
    real_images = torch.randn(2, 3, 256, 256).to(device)
    real_scores = discriminator(real_images)
    fake_scores = discriminator(fake_images.detach())
    
    print(f"Real scores shape: {real_scores.shape}")
    print(f"Fake scores shape: {fake_scores.shape}")
    
    print("Model tests passed!")


if __name__ == "__main__":
    test_models()
