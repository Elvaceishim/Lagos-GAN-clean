"""
LagosGAN Demo Application

Interactive Gradio demo showcasing both AfroCover and Lagos2Duplex models.
"""

import os
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import torch
from PIL import Image

from gradio_client import utils as gradio_client_utils

try:
    from afrocover.models import StyleGAN2Generator
except ModuleNotFoundError:  # pragma: no cover - deployed Spaces copy omits training code
    StyleGAN2Generator = None
    print("Warning: afrocover.models not available; demo will use AfroCover placeholders.")

try:
    from lagos2duplex.models import CycleGANGenerator, get_norm_layer
except ModuleNotFoundError:  # pragma: no cover - deployed Spaces copy omits training code
    CycleGANGenerator = None

    def get_norm_layer(*_args, **_kwargs):  # type: ignore
        raise RuntimeError("lagos2duplex.models not available; cannot build generator.")


# Temporary monkey patch: Gradio 4.** can emit boolean values inside JSON
# schemas (e.g. `additionalProperties: False`). The stock conversion helper
# assumes dictionaries and crashes when it encounters those booleans while
# building the API docs. This shim coalesces boolean schemas into reasonable
# string representations so the demo can launch until the upstream bug is
# resolved.
_ORIGINAL_JSON_SCHEMA_TO_PYTHON_TYPE = gradio_client_utils._json_schema_to_python_type


def _json_schema_to_python_type_safe(schema: Any, defs) -> str:
    if isinstance(schema, bool):
        return "Any" if schema else "None"
    return _ORIGINAL_JSON_SCHEMA_TO_PYTHON_TYPE(schema, defs)


def _json_schema_to_python_type_public(schema: Any) -> str:
    if isinstance(schema, bool):
        return "Any" if schema else "None"
    defs = schema.get("$defs") if isinstance(schema, dict) else None
    type_hint = _json_schema_to_python_type_safe(schema, defs)
    return type_hint.replace(
        gradio_client_utils.CURRENT_FILE_DATA_FORMAT, "filepath"
    )


gradio_client_utils._json_schema_to_python_type = _json_schema_to_python_type_safe
gradio_client_utils.json_schema_to_python_type = _json_schema_to_python_type_public

from afrocover.models import StyleGAN2Generator
from lagos2duplex.models import CycleGANGenerator, get_norm_layer


class LagosGANDemo:
    """Main demo class that handles both AfroCover and Lagos2Duplex models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model paths
        self.afrocover_model_path = (
            "models/afrocover/latest.pt"
        )
        self.lagos2duplex_model_path = (
            "models/lagos2duplex/latest.pt"
        )
        self.lagos2duplex_model_path = self._resolve_checkpoint(
            [
                "checkpoints/lagos2duplex/final_model.pt",
                "checkpoints/lagos2duplex/latest.pt",
            ]
        )

        self.afrocover_z_dim = 512
        self.afrocover_image_size = 256
        self.afrocover_channel_multiplier = 1.0

        self.lagos_input_nc = 3
        self.lagos_output_nc = 3
        self.lagos_image_size = 256
        
        # Load models
        self.afrocover_model = self._load_afrocover_model()
        self.lagos2duplex_model = self._load_lagos2duplex_model()
        
        print("LagosGAN Demo initialized!")
    
    def _resolve_checkpoint(self, candidates):
        for p in candidates:
            if p and os.path.exists(p):
                return p
        return None

    def _load_afrocover_model(self):
        """Load the trained AfroCover StyleGAN2 model"""
        if StyleGAN2Generator is None:
            print("AfroCover code unavailable; demo will show placeholders.")
            return None
        if not self.afrocover_model_path:
            print("AfroCover checkpoint not found; demo will use placeholders")
            return None

        try:
            print(f"Loading AfroCover model from {self.afrocover_model_path}...")
            checkpoint = torch.load(self.afrocover_model_path, map_location=self.device)
            cfg = checkpoint.get("args", {})
            self.afrocover_z_dim = cfg.get("z_dim", 512)
            self.afrocover_image_size = cfg.get("image_size", 256)
            self.afrocover_channel_multiplier = cfg.get("channel_multiplier", 1.0)

            generator = StyleGAN2Generator(
                z_dim=self.afrocover_z_dim,
                w_dim=self.afrocover_z_dim,
                img_resolution=self.afrocover_image_size,
                img_channels=3,
                channel_multiplier=self.afrocover_channel_multiplier,
            ).to(self.device)

            state = checkpoint.get("generator_state_dict") or checkpoint.get("generator")
            if state is None:
                raise ValueError("Generator weights not found in checkpoint")

            generator.load_state_dict(state, strict=False)
            generator.eval()
            return generator
        except Exception as e:
            print(f"Error loading AfroCover model: {e}")
            return None

    def _load_lagos2duplex_model(self):
        """Load the trained Lagos2Duplex CycleGAN model"""
        if CycleGANGenerator is None:
            print("Lagos2Duplex code unavailable; demo will show placeholders.")
            return None
        if not self.lagos2duplex_model_path:
            print("Lagos2Duplex checkpoint not found; demo will use placeholders")
            return None

        try:
            print(f"Loading Lagos2Duplex model from {self.lagos2duplex_model_path}...")
            checkpoint = torch.load(self.lagos2duplex_model_path, map_location=self.device)
            cfg = checkpoint.get("config", {})
            model_cfg = cfg.get("model", {})

            self.lagos_input_nc = model_cfg.get("input_nc", 3)
            self.lagos_output_nc = model_cfg.get("output_nc", 3)
            ngf = model_cfg.get("ngf", 64)
            norm = model_cfg.get("norm", "instance")
            use_dropout = model_cfg.get("use_dropout", False)
            n_blocks = model_cfg.get("n_blocks", 9)

            data_cfg = cfg.get("data", {})
            self.lagos_image_size = data_cfg.get("image_size", 256)

            norm_layer = get_norm_layer(norm)

            generator = CycleGANGenerator(
                input_nc=self.lagos_input_nc,
                output_nc=self.lagos_output_nc,
                ngf=ngf,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                n_blocks=n_blocks,
            ).to(self.device)

            state = checkpoint.get("G_AB_state_dict")
            if state is None:
                raise ValueError("G_AB_state_dict not found in checkpoint")

            generator.load_state_dict(state, strict=False)
            generator.eval()
            return generator
        except Exception as e:
            print(f"Error loading Lagos2Duplex model: {e}")
            return None
    
    def generate_album_cover(self, style_seed=None, num_images=4):
        """Generate African-inspired album covers"""
        if self.afrocover_model is None:
            return self._create_placeholder_images(num_images, "AfroCover model not loaded")

        try:
            print(f"Generating {num_images} album covers with seed {style_seed}")
            if style_seed is not None:
                torch.manual_seed(int(style_seed))
            z = torch.randn(num_images, self.afrocover_z_dim, device=self.device)
            with torch.no_grad():
                output = self.afrocover_model(z)

            images = []
            for img_tensor in output:
                img_tensor = img_tensor.clamp(-1, 1)
                img_tensor = (img_tensor + 1.0) / 2.0  # to [0,1]
                img_array = (img_tensor.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
                images.append(Image.fromarray(img_array))
            return images
        except Exception as e:
            print(f"Error generating album covers: {e}")
            return self._create_placeholder_images(num_images, f"Error: {e}")
    
    def transform_house(self, input_image):
        """Transform Lagos house to modern duplex"""
        if self.lagos2duplex_model is None:
            return self._create_placeholder_image("Lagos2Duplex model not loaded")

        if input_image is None:
            return self._create_placeholder_image("Please upload an image")

        try:
            print("Transforming house to duplex...")
            img = Image.fromarray(input_image).convert("RGB")
            img = img.resize((self.lagos_image_size, self.lagos_image_size), Image.BICUBIC)
            img_tensor = torch.from_numpy(np.asarray(img)).float() / 255.0
            img_tensor = (img_tensor - 0.5) / 0.5  # [-1,1]
            img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.lagos2duplex_model(img_tensor)

            output = output[0].clamp(-1, 1)
            output = (output + 1.0) / 2.0
            output_array = (output.detach().cpu().permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            return Image.fromarray(output_array)
        except Exception as e:
            print(f"Error transforming house: {e}")
            return self._create_placeholder_image(f"Error: {e}")
    
    def _create_placeholder_image(self, text="Placeholder"):
        """Create a placeholder image with text"""
        img = Image.new('RGB', (256, 256), color='lightgray')
        return img
    
    def _create_placeholder_images(self, num_images, text="Placeholder"):
        """Create multiple placeholder images"""
        return [self._create_placeholder_image(text) for _ in range(num_images)]


def create_afrocover_interface(demo):
    """Create the AfroCover tab interface"""
    with gr.Tab("🎨 AfroCover - Album Art Generator"):
        gr.Markdown("""
        # AfroCover: African-Inspired Album Cover Generator
        
        Generate unique album covers inspired by African art, patterns, and aesthetics using StyleGAN2.
        """)
        
        with gr.Row():
            with gr.Column():
                style_seed = gr.Number(
                    label="Style Seed (optional)", 
                    value=None, 
                    precision=0,
                    info="Set a seed for reproducible results"
                )
                num_images = gr.Slider(
                    minimum=1, 
                    maximum=8, 
                    value=4, 
                    step=1, 
                    label="Number of Covers"
                )
                generate_btn = gr.Button("🎨 Generate Album Covers", variant="primary")
            
            with gr.Column():
                output_gallery = gr.Gallery(
                    label="Generated Album Covers",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    rows=2,
                    height="auto"
                )
        
        # Event handlers
        generate_btn.click(
            fn=demo.generate_album_cover,
            inputs=[style_seed, num_images],
            outputs=output_gallery
        )
        
        gr.Markdown("""
        ### Tips:
        - Try different seeds to explore various artistic styles
        - Generated covers are at 256×256 resolution
        - Each generation creates unique African-inspired designs
        """)


def create_lagos2duplex_interface(demo):
    """Create the Lagos2Duplex tab interface"""
    with gr.Tab("🏠 Lagos2Duplex - House Transformer"):
        gr.Markdown("""
        # Lagos2Duplex: Old Lagos Houses → Modern Duplexes
        
        Transform old Lagos house photos into modern duplex designs using CycleGAN.
        """)
        
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(
                    label="Upload Lagos House Photo",
                    type="numpy",
                    height=300
                )
                transform_btn = gr.Button("🏗️ Transform to Duplex", variant="primary")
                
                gr.Markdown("""
                ### Upload Guidelines:
                - Use clear photos of Lagos houses
                - Front-facing views work best
                - Avoid heavily obstructed views
                """)
            
            with gr.Column():
                output_image = gr.Image(
                    label="Modern Duplex Design",
                    height=300
                )
                
                gr.Markdown("""
                ### About the Transformation:
                - AI generates conceptual duplex designs
                - Results are for inspiration only
                - Not suitable for actual construction
                """)
        
        # Event handlers
        transform_btn.click(
            fn=demo.transform_house,
            inputs=input_image,
            outputs=output_image
        )


def create_demo():
    """Create the main Gradio demo interface"""
    # Initialize demo
    demo_instance = LagosGANDemo()
    
    # Create Gradio app
    with gr.Blocks(
        title="LagosGAN Demo",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        #gallery {
            min-height: 400px;
        }
        """
    ) as demo:
        
        gr.Markdown("""
        # LagosGAN: AI-Powered African Creativity
        
        Explore two innovative GAN applications celebrating African design and architecture.
        """)
        
        # Create tabs
        create_afrocover_interface(demo_instance)
        create_lagos2duplex_interface(demo_instance)
        
        # Footer
        with gr.Tab("ℹ️ About"):
            gr.Markdown("""
            ## About LagosGAN
            
            LagosGAN is an experimental ML project showcasing how GANs can power African-centered synthetic creativity.
            
            ### AfroCover (StyleGAN2)
            - Generates African-inspired music/album cover art
            - Trained on curated African design elements
            - 256×256 resolution outputs
            
            ### Lagos2Duplex (CycleGAN)  
            - Transforms old Lagos house photos into modern duplex mockups
            - Explores architectural evolution in Nigeria
            - Conceptual designs for inspiration
            
            ### Ethics & Limitations
            - All datasets are ethically sourced and properly licensed
            - Generated content is for creative inspiration only
            - Architectural designs are not construction-ready
            - Results may contain biases present in training data
            
            ### Links
            - [GitHub Repository](#)
            - [Research Article](#)
            - [Dataset Information](docs/dataset_cards.md)
            - [Model Documentation](docs/model_cards.md)
            
            ---
            *Built with PyTorch, StyleGAN2-ADA, CycleGAN, and Gradio*
            """)
    
    return demo


def main():
    """Main function to launch the demo"""
    print("Starting LagosGAN Demo...")
    
    # Create and launch demo
    demo = create_demo()
    
    # Launch with appropriate settings
    demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
    debug=True,
    show_error=True,
)




if __name__ == "__main__":
    main()
