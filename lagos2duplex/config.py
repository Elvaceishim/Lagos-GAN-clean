"""
Training Configuration for Lagos2Duplex CycleGAN

This file contains all configuration parameters for training the CycleGAN model.
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import torch


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    # Generator parameters
    input_nc: int = 3               # Input channels (RGB)
    output_nc: int = 3              # Output channels (RGB)
    ngf: int = 64                   # Number of generator filters in first conv layer
    ndf: int = 64                   # Number of discriminator filters in first conv layer
    netG: str = 'resnet_9blocks'    # Generator architecture ('resnet_9blocks', 'resnet_6blocks')
    netD: str = 'basic'             # Discriminator architecture ('basic', 'n_layers')
    n_layers_D: int = 3             # Number of discriminator layers (only used if netD='n_layers')
    norm: str = 'instance'          # Normalization layer ('batch', 'instance', 'none')
    use_dropout: bool = False       # Use dropout in generator
    init_type: str = 'normal'       # Network initialization ('normal', 'xavier', 'kaiming', 'orthogonal')
    init_gain: float = 0.02         # Scaling factor for initialization


@dataclass
class TrainingConfig:
    """Training hyperparameters"""
    # Training settings
    num_epochs: int = 200           # Number of training epochs
    batch_size: int = 1             # Batch size
    learning_rate: float = 0.0002   # Learning rate for Adam optimizer
    beta1: float = 0.5              # Beta1 for Adam optimizer
    beta2: float = 0.999            # Beta2 for Adam optimizer
    
    # Loss weights
    lambda_cycle: float = 10.0      # Weight for cycle consistency loss
    lambda_identity: float = 0.5    # Weight for identity loss (0 to disable)
    lambda_perceptual: float = 0.0  # Weight for perceptual loss (0 to disable)
    
    # Loss types
    gan_mode: str = 'lsgan'         # GAN loss type ('lsgan', 'vanilla', 'wgangp')
    cycle_loss_type: str = 'l1'     # Cycle loss type ('l1', 'l2')
    identity_loss_type: str = 'l1'  # Identity loss type ('l1', 'l2')
    
    # Learning rate scheduling
    lr_policy: str = 'linear'       # Learning rate policy ('linear', 'step', 'plateau', 'cosine')
    lr_decay_iters: int = 50        # Multiply by a gamma every lr_decay_iters iterations
    n_epochs_decay: int = 100       # Number of epochs to linearly decay learning rate to zero
    
    # Image pool
    pool_size: int = 50             # Size of image buffer for discriminator training
    
    # Gradient clipping
    max_grad_norm: Optional[float] = None  # Maximum gradient norm for clipping


@dataclass
class DataConfig:
    """Data loading configuration"""
    # Dataset paths (relative to project root)
    data_path: str = "data_processed"  # Path to processed dataset
    image_size: int = 256              # Image resolution
    
    # Data loading
    num_workers: int = 4               # Number of data loading workers
    pin_memory: bool = True            # Pin memory for faster GPU transfer
    
    # Data augmentation
    augmentations: bool = True         # Enable data augmentations for training
    paired: bool = False               # Whether to use paired dataset (if available)


@dataclass
class LoggingConfig:
    """Logging and visualization configuration"""
    # Checkpoints
    save_epoch_freq: int = 20          # Frequency of saving checkpoints
    save_latest_freq: int = 5000       # Frequency of saving latest model (iterations)
    
    # Logging
    print_freq: int = 100              # Frequency of printing training status
    display_freq: int = 400            # Frequency of showing training images
    
    # Weights & Biases
    use_wandb: bool = True             # Enable Weights & Biases logging
    wandb_project: str = "lagosgan-lagos2duplex"  # W&B project name
    wandb_run_name: Optional[str] = None          # W&B run name (auto-generated if None)
    
    # TensorBoard
    use_tensorboard: bool = False      # Enable TensorBoard logging
    
    # Sample generation
    num_test_samples: int = 4          # Number of test samples to generate

    # Evaluation
    eval_fid_freq: int = 0            # Frequency (in epochs) to compute FID during training (0 disables)


@dataclass
class PathConfig:
    """File system paths configuration"""
    # Base directories
    project_root: str = "/Users/mac/Lagos-GAN"
    checkpoints_dir: str = "checkpoints/lagos2duplex"
    results_dir: str = "results/lagos2duplex"
    logs_dir: str = "logs/lagos2duplex"
    
    # Specific files
    resume_checkpoint: Optional[str] = None  # Path to checkpoint to resume from
    
    def __post_init__(self):
        """Convert relative paths to absolute paths"""
        self.checkpoints_dir = os.path.join(self.project_root, self.checkpoints_dir)
        self.results_dir = os.path.join(self.project_root, self.results_dir)
        self.logs_dir = os.path.join(self.project_root, self.logs_dir)
        
        # Create directories if they don't exist
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)


@dataclass
class SystemConfig:
    """System and hardware configuration"""
    # Device settings
    device: str = "auto"               # Device to use ('auto', 'cuda', 'cpu')
    gpu_ids: list = None               # GPU IDs to use (None for single GPU)
    
    # Performance
    mixed_precision: bool = False      # Enable mixed precision training
    compile_model: bool = False        # Enable torch.compile (PyTorch 2.0+)
    
    # Reproducibility
    seed: Optional[int] = 42           # Random seed for reproducibility
    deterministic: bool = False        # Enable deterministic operations
    
    def __post_init__(self):
        """Auto-configure device settings"""
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if self.gpu_ids is None and self.device == "cuda":
            self.gpu_ids = [0]  # Use first GPU by default


class CycleGANConfig:
    """Main configuration class combining all sub-configurations"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration with optional overrides
        
        Args:
            config_dict: Dictionary of configuration overrides
        """
        # Initialize sub-configurations
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.data = DataConfig()
        self.logging = LoggingConfig()
        self.paths = PathConfig()
        self.system = SystemConfig()
        
        # Apply overrides if provided
        if config_dict:
            self.update_from_dict(config_dict)
        
        # Set derived paths
        self.data.data_path = os.path.join(self.paths.project_root, self.data.data_path)
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for section_name, section_config in config_dict.items():
            if hasattr(self, section_name):
                section = getattr(self, section_name)
                for key, value in section_config.items():
                    if hasattr(section, key):
                        setattr(section, key, value)
                    else:
                        print(f"Warning: Unknown config key {section_name}.{key}")
            else:
                print(f"Warning: Unknown config section {section_name}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'data': self.data.__dict__,
            'logging': self.logging.__dict__,
            'paths': self.paths.__dict__,
            'system': self.system.__dict__
        }
    
    def save(self, filepath: str):
        """Save configuration to file"""
        import json
        config_dict = self.to_dict()
        
        # Convert non-serializable types
        def convert_value(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif obj is None:
                return None
            return obj
        
        def convert_dict(d):
            return {k: convert_value(v) if not isinstance(v, dict) else convert_dict(v) 
                   for k, v in d.items()}
        
        config_dict = convert_dict(config_dict)
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        print(f"Configuration saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str):
        """Load configuration from file"""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        return cls(config_dict)
    
    def validate(self):
        """Validate configuration parameters"""
        # Check required paths exist
        if not os.path.exists(self.data.data_path):
            raise ValueError(f"Data path does not exist: {self.data.data_path}")
        
        # Check device availability
        if self.system.device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available, falling back to CPU")
            self.system.device = "cpu"
        
        # Check parameter ranges
        assert self.training.batch_size > 0, "Batch size must be positive"
        assert self.training.learning_rate > 0, "Learning rate must be positive"
        assert self.training.lambda_cycle >= 0, "Lambda cycle must be non-negative"
        assert self.training.lambda_identity >= 0, "Lambda identity must be non-negative"
        assert self.data.image_size > 0, "Image size must be positive"
        
        print("Configuration validation passed")


def get_default_config() -> CycleGANConfig:
    """Get default configuration for Lagos2Duplex training"""
    return CycleGANConfig()


def get_quick_test_config() -> CycleGANConfig:
    """Get configuration for quick testing/debugging"""
    config_overrides = {
        'training': {
            'num_epochs': 5,
            'batch_size': 2,
            'save_epoch_freq': 2,
            'print_freq': 10
        },
        'data': {
            'num_workers': 2
        },
        'logging': {
            'use_wandb': False,
            'display_freq': 50
        }
    }
    return CycleGANConfig(config_overrides)


def get_high_quality_config() -> CycleGANConfig:
    """Get configuration for high-quality training"""
    config_overrides = {
        'model': {
            'netG': 'resnet_9blocks',
            'ngf': 64,
            'ndf': 64
        },
        'training': {
            'num_epochs': 300,
            'lambda_cycle': 15.0,
            'lambda_identity': 1.0,
            'lambda_perceptual': 1.0
        },
        'data': {
            'image_size': 256,
            'augmentations': True
        }
    }
    return CycleGANConfig(config_overrides)


if __name__ == "__main__":
    # Test configuration
    print("Testing CycleGAN Configuration...")
    
    # Test default config
    config = get_default_config()
    config.validate()
    print("Default configuration created successfully")
    
    # Test saving and loading
    config_path = "/tmp/test_config.json"
    config.save(config_path)
    loaded_config = CycleGANConfig.load(config_path)
    print("Configuration save/load test passed")
    
    # Test different configurations
    quick_config = get_quick_test_config()
    high_quality_config = get_high_quality_config()
    
    print(f"Quick test config epochs: {quick_config.training.num_epochs}")
    print(f"High quality config epochs: {high_quality_config.training.num_epochs}")
    
    print("Configuration tests passed!")
