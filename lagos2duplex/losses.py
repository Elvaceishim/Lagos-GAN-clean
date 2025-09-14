"""
CycleGAN Loss Functions for Lagos2Duplex

Implementation of various loss functions used in CycleGAN training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GANLoss(nn.Module):
    """GAN loss function with support for different loss types"""
    
    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        """
        Args:
            gan_mode (str): Type of GAN loss ('lsgan', 'vanilla', 'wgangp')
            target_real_label (float): Label for real images
            target_fake_label (float): Label for fake images
        """
        super().__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')
    
    def get_target_tensor(self, prediction, target_is_real):
        """Create a tensor with target labels"""
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)
    
    def __call__(self, prediction, target_is_real):
        """Calculate loss given discriminator output and ground truth"""
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        
        return loss


class CycleLoss(nn.Module):
    """Cycle consistency loss for CycleGAN"""
    
    def __init__(self, loss_type='l1'):
        """
        Args:
            loss_type (str): Type of reconstruction loss ('l1' or 'l2')
        """
        super().__init__()
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError(f'Loss type {loss_type} not implemented')
    
    def __call__(self, pred, target):
        return self.loss(pred, target)


class IdentityLoss(nn.Module):
    """Identity loss for CycleGAN"""
    
    def __init__(self, loss_type='l1'):
        """
        Args:
            loss_type (str): Type of identity loss ('l1' or 'l2')
        """
        super().__init__()
        if loss_type == 'l1':
            self.loss = nn.L1Loss()
        elif loss_type == 'l2':
            self.loss = nn.MSELoss()
        else:
            raise NotImplementedError(f'Loss type {loss_type} not implemented')
    
    def __call__(self, pred, target):
        return self.loss(pred, target)


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features for better image quality"""
    
    def __init__(self, layer_weights=None):
        super().__init__()
        import torchvision.models as models
        
        # Load pre-trained VGG16
        vgg = models.vgg16(pretrained=True).features
        self.vgg = vgg.eval()
        
        # Freeze VGG parameters
        for param in self.vgg.parameters():
            param.requires_grad = False
        
        # Define which layers to use for loss computation
        self.layer_weights = layer_weights or {
            '4': 1.0,   # relu1_2
            '9': 1.0,   # relu2_2
            '16': 1.0,  # relu3_3
            '23': 1.0,  # relu4_3
        }
        
        self.loss = nn.L1Loss()
    
    def forward(self, input, target):
        """Compute perceptual loss between input and target"""
        input_features = self.extract_features(input)
        target_features = self.extract_features(target)
        
        loss = 0
        for layer in self.layer_weights:
            loss += self.layer_weights[layer] * self.loss(
                input_features[layer], 
                target_features[layer]
            )
        
        return loss
    
    def extract_features(self, x):
        """Extract features from different VGG layers"""
        features = {}
        for i, layer in enumerate(self.vgg):
            x = layer(x)
            if str(i) in self.layer_weights:
                features[str(i)] = x
        return features


class CycleGANLoss:
    """Combined loss function for CycleGAN training"""
    
    def __init__(self, 
                 gan_mode='lsgan',
                 lambda_cycle=10.0,
                 lambda_identity=0.5,
                 lambda_perceptual=0.0,
                 cycle_loss_type='l1',
                 identity_loss_type='l1'):
        """
        Args:
            gan_mode (str): Type of GAN loss
            lambda_cycle (float): Weight for cycle consistency loss
            lambda_identity (float): Weight for identity loss
            lambda_perceptual (float): Weight for perceptual loss
            cycle_loss_type (str): Type of cycle loss
            identity_loss_type (str): Type of identity loss
        """
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        self.lambda_perceptual = lambda_perceptual
        
        # Initialize loss functions
        self.gan_loss = GANLoss(gan_mode)
        self.cycle_loss = CycleLoss(cycle_loss_type)
        self.identity_loss = IdentityLoss(identity_loss_type)
        
        if lambda_perceptual > 0:
            self.perceptual_loss = PerceptualLoss()
        else:
            self.perceptual_loss = None
    
    def compute_generator_loss(self, real_A, real_B, fake_A, fake_B, 
                             rec_A, rec_B, idt_A, idt_B, 
                             D_A_fake, D_B_fake):
        """Compute total generator loss"""
        loss_dict = {}
        
        # Adversarial losses
        loss_G_A = self.gan_loss(D_A_fake, True)
        loss_G_B = self.gan_loss(D_B_fake, True)
        loss_dict['G_A'] = loss_G_A
        loss_dict['G_B'] = loss_G_B
        
        # Cycle consistency losses
        loss_cycle_A = self.cycle_loss(rec_A, real_A) * self.lambda_cycle
        loss_cycle_B = self.cycle_loss(rec_B, real_B) * self.lambda_cycle
        loss_dict['cycle_A'] = loss_cycle_A
        loss_dict['cycle_B'] = loss_cycle_B
        
        # Identity losses
        loss_identity = 0
        if self.lambda_identity > 0 and idt_A is not None and idt_B is not None:
            loss_idt_A = self.identity_loss(idt_A, real_A) * self.lambda_identity
            loss_idt_B = self.identity_loss(idt_B, real_B) * self.lambda_identity
            loss_identity = loss_idt_A + loss_idt_B
            loss_dict['identity_A'] = loss_idt_A
            loss_dict['identity_B'] = loss_idt_B
        
        # Perceptual losses
        loss_perceptual = 0
        if self.perceptual_loss is not None and self.lambda_perceptual > 0:
            loss_perc_A = self.perceptual_loss(fake_A, real_A) * self.lambda_perceptual
            loss_perc_B = self.perceptual_loss(fake_B, real_B) * self.lambda_perceptual
            loss_perceptual = loss_perc_A + loss_perc_B
            loss_dict['perceptual_A'] = loss_perc_A
            loss_dict['perceptual_B'] = loss_perc_B
        
        # Total generator loss
        total_loss = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_identity + loss_perceptual
        loss_dict['total'] = total_loss
        
        return total_loss, loss_dict
    
    def compute_discriminator_loss(self, D_real, D_fake):
        """Compute discriminator loss"""
        loss_real = self.gan_loss(D_real, True)
        loss_fake = self.gan_loss(D_fake, False)
        loss_D = (loss_real + loss_fake) * 0.5
        
        return loss_D, {'real': loss_real, 'fake': loss_fake, 'total': loss_D}


def test_losses():
    """Test loss functions"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy data
    batch_size = 2
    channels = 3
    height = width = 256
    
    real_A = torch.randn(batch_size, channels, height, width).to(device)
    real_B = torch.randn(batch_size, channels, height, width).to(device)
    fake_A = torch.randn(batch_size, channels, height, width).to(device)
    fake_B = torch.randn(batch_size, channels, height, width).to(device)
    rec_A = torch.randn(batch_size, channels, height, width).to(device)
    rec_B = torch.randn(batch_size, channels, height, width).to(device)
    
    # Test individual losses
    print("Testing GAN Loss...")
    gan_loss = GANLoss('lsgan').to(device)
    D_output = torch.randn(batch_size, 1, 30, 30).to(device)
    loss_real = gan_loss(D_output, True)
    loss_fake = gan_loss(D_output, False)
    print(f"GAN loss real: {loss_real.item():.4f}")
    print(f"GAN loss fake: {loss_fake.item():.4f}")
    
    print("Testing Cycle Loss...")
    cycle_loss = CycleLoss('l1').to(device)
    cycle_loss_value = cycle_loss(rec_A, real_A)
    print(f"Cycle loss: {cycle_loss_value.item():.4f}")
    
    print("Testing Combined CycleGAN Loss...")
    cyclegan_loss = CycleGANLoss()
    # Move loss functions to device if needed
    cyclegan_loss.gan_loss = cyclegan_loss.gan_loss.to(device)
    cyclegan_loss.cycle_loss = cyclegan_loss.cycle_loss.to(device)
    cyclegan_loss.identity_loss = cyclegan_loss.identity_loss.to(device)
    
    D_A_fake = torch.randn(batch_size, 1, 30, 30).to(device)
    D_B_fake = torch.randn(batch_size, 1, 30, 30).to(device)
    
    total_loss, loss_dict = cyclegan_loss.compute_generator_loss(
        real_A, real_B, fake_A, fake_B, rec_A, rec_B, None, None, D_A_fake, D_B_fake
    )
    
    print(f"Total generator loss: {total_loss.item():.4f}")
    for key, value in loss_dict.items():
        if key != 'total':
            print(f"  {key}: {value.item():.4f}")
    
    print("Loss tests passed!")


if __name__ == "__main__":
    test_losses()
