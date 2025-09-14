#!/usr/bin/env python3
"""
Lagos-GAN Demo Runner

Script to easily launch the Lagos-GAN Gradio demo.
"""

import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Launch the Lagos-GAN demo"""
    try:
        print("🚀 Starting Lagos-GAN Demo...")
        print("Loading demo application...")
        
        from demo.app import LagosGANDemo
        
        # Create demo instance
        demo_instance = LagosGANDemo()
        
        print("✅ Demo loaded successfully!")
        print("🌐 Starting Gradio interface...")
        
        # Note: The actual Gradio interface launch code would go here
        # For now, we just show that the demo can be instantiated
        print("📝 Demo is ready! (Interface implementation pending)")
        print("💡 You can now integrate this with a Gradio interface.")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error starting demo: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
