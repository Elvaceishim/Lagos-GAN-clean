"""
LagosGAN Demo Application

Interactive Gradio demo showcasing both AfroCover and Lagos2Duplex models.
"""

import gradio as gr
import torch
import numpy as np
from PIL import Image
import os
from pathlib import Path

# Import our models (once implemented)
# from afrocover.models import StyleGAN2Generator
# from lagos2duplex.models import CycleGANGenerator


class LagosGANDemo:
    """Main demo class that handles both AfroCover and Lagos2Duplex models"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Model paths
        self.afrocover_model_path = "checkpoints/afrocover/final_model.pt"
        self.lagos2duplex_model_path = "checkpoints/lagos2duplex/final_model.pt"
        
        # Load models
        self.afrocover_model = self._load_afrocover_model()
        self.lagos2duplex_model = self._load_lagos2duplex_model()
        
        print("LagosGAN Demo initialized!")
    
    def _load_afrocover_model(self):
        """Load the trained AfroCover StyleGAN2 model"""
        try:
            # TODO: Implement model loading
            print("Loading AfroCover model...")
            # model = StyleGAN2Generator(...)
            # if os.path.exists(self.afrocover_model_path):
            #     checkpoint = torch.load(self.afrocover_model_path, map_location=self.device)
            #     model.load_state_dict(checkpoint['generator_state_dict'])
            # model.eval()
            # return model
            return None
        except Exception as e:
            print(f"Error loading AfroCover model: {e}")
            return None
    
    def _load_lagos2duplex_model(self):
        """Load the trained Lagos2Duplex CycleGAN model"""
        try:
            # TODO: Implement model loading
            print("Loading Lagos2Duplex model...")
            # model = CycleGANGenerator(...)
            # if os.path.exists(self.lagos2duplex_model_path):
            #     checkpoint = torch.load(self.lagos2duplex_model_path, map_location=self.device)
            #     model.load_state_dict(checkpoint['G_AB_state_dict'])
            # model.eval()
            # return model
            return None
        except Exception as e:
            print(f"Error loading Lagos2Duplex model: {e}")
            return None
    
    def generate_album_cover(self, style_seed=None, num_images=4):
        """Generate African-inspired album covers"""
        if self.afrocover_model is None:
            return self._create_placeholder_images(num_images, "AfroCover model not loaded")
        
        try:
            # TODO: Implement generation
            print(f"Generating {num_images} album covers with seed {style_seed}")
            
            # Placeholder implementation
            # if style_seed is not None:
            #     torch.manual_seed(style_seed)
            # 
            # with torch.no_grad():
            #     z = torch.randn(num_images, 512).to(self.device)
            #     generated_images = self.afrocover_model(z)
            #     
            #     # Convert to PIL images
            #     images = []
            #     for i in range(num_images):
            #         img_tensor = generated_images[i]
            #         img_tensor = (img_tensor + 1) / 2  # Denormalize from [-1, 1] to [0, 1]
            #         img_array = img_tensor.cpu().numpy().transpose(1, 2, 0)
            #         img_array = (img_array * 255).astype(np.uint8)
            #         images.append(Image.fromarray(img_array))
            #     
            #     return images
            
            return self._create_placeholder_images(num_images, "Generated Album Covers")
            
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
            # TODO: Implement transformation
            print("Transforming house to duplex...")
            
            # Placeholder implementation
            # # Preprocess input image
            # img = Image.fromarray(input_image).convert('RGB')
            # img = img.resize((256, 256))
            # img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()
            # img_tensor = (img_tensor / 255.0 - 0.5) / 0.5  # Normalize to [-1, 1]
            # img_tensor = img_tensor.unsqueeze(0).to(self.device)
            # 
            # with torch.no_grad():
            #     transformed = self.lagos2duplex_model(img_tensor)
            #     
            #     # Convert back to PIL image
            #     output_tensor = transformed[0]
            #     output_tensor = (output_tensor + 1) / 2  # Denormalize
            #     output_array = output_tensor.cpu().numpy().transpose(1, 2, 0)
            #     output_array = (output_array * 255).astype(np.uint8)
            #     
            #     return Image.fromarray(output_array)
            
            return self._create_placeholder_image("Transformed Duplex")
            
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
        # 🇳🇬 LagosGAN: AI-Powered African Creativity
        
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
            
            ### 🎨 AfroCover (StyleGAN2)
            - Generates African-inspired music/album cover art
            - Trained on curated African design elements
            - 256×256 resolution outputs
            
            ### 🏠 Lagos2Duplex (CycleGAN)  
            - Transforms old Lagos house photos into modern duplex mockups
            - Explores architectural evolution in Nigeria
            - Conceptual designs for inspiration
            
            ### ⚖️ Ethics & Limitations
            - All datasets are ethically sourced and properly licensed
            - Generated content is for creative inspiration only
            - Architectural designs are not construction-ready
            - Results may contain biases present in training data
            
            ### 🔗 Links
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
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True for public sharing
        debug=True,             # Enable debug mode
        show_error=True         # Show errors in interface
    )


if __name__ == "__main__":
    main()
