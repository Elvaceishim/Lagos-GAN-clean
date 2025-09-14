"""
CycleGAN Models for Lagos2Duplex

Implementation of CycleGAN Generator and Discriminator for house transformation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import random


class ResnetBlock(nn.Module):
    """Resnet Block for CycleGAN Generator"""
    
    def __init__(self, dim, norm_layer, use_dropout=False, use_bias=False):
        super().__init__()
        
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        
        conv_block += [nn.ReflectionPad2d(1)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, bias=use_bias), norm_layer(dim)]
        
        self.conv_block = nn.Sequential(*conv_block)
    
    def forward(self, x):
        return x + self.conv_block(x)


class CycleGANGenerator(nn.Module):
    """CycleGAN Generator with ResNet backbone"""
    
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d, 
                 use_dropout=False, n_blocks=9):
        super().__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        # Initial convolution block
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        
        # Downsampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
        
        # ResNet blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        
        # Upsampling
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        
        # Final convolution
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7)]
        model += [nn.Tanh()]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        return self.model(input)


class CycleGANDiscriminator(nn.Module):
    """PatchGAN Discriminator for CycleGAN"""
    
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, input):
        return self.model(input)


class ImagePool:
    """
    Buffer that stores previously generated images for discriminator training.
    This improves GAN training stability.
    """
    
    def __init__(self, pool_size=50):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []
    
    def query(self, images):
        if self.pool_size == 0:
            return images
        
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size - 1)
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    return_images.append(image)
        
        return_images = torch.cat(return_images, 0)
        return return_images


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer"""
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return nn.Identity()
    else:
        raise NotImplementedError(f'normalization layer [{norm_type}] is not found')
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights"""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    
    net.apply(init_func)


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a generator"""
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    
    if netG == 'resnet_9blocks':
        net = CycleGANGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'resnet_6blocks':
        net = CycleGANGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6)
    else:
        raise NotImplementedError(f'Generator model name [{netG}] is not recognized')
    
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Create a discriminator"""
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    
    if netD == 'basic':
        net = CycleGANDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    elif netD == 'n_layers':
        net = CycleGANDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer)
    else:
        raise NotImplementedError(f'Discriminator model name [{netD}] is not recognized')
    
    return init_net(net, init_type, init_gain, gpu_ids)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network"""
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, init_gain=init_gain)
    return net


def test_models():
    """Test model creation and forward pass"""
    import random
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test Generator
    print("Testing CycleGAN Generator...")
    generator = CycleGANGenerator(
        input_nc=3,
        output_nc=3,
        ngf=64,
        norm_layer=nn.InstanceNorm2d,
        n_blocks=9
    ).to(device)
    
    test_input = torch.randn(2, 3, 256, 256).to(device)
    output = generator(test_input)
    print(f"Generator output shape: {output.shape}")
    
    # Test Discriminator
    print("Testing CycleGAN Discriminator...")
    discriminator = CycleGANDiscriminator(
        input_nc=3,
        ndf=64,
        norm_layer=nn.InstanceNorm2d
    ).to(device)
    
    disc_output = discriminator(test_input)
    print(f"Discriminator output shape: {disc_output.shape}")
    
    # Test Image Pool
    print("Testing Image Pool...")
    pool = ImagePool(pool_size=50)
    pooled_images = pool.query(output)
    print(f"Pooled images shape: {pooled_images.shape}")
    
    print("Model tests passed!")


if __name__ == "__main__":
    test_models()
