#!/usr/bin/env python3
"""
Lagos-GAN Environment Test Script

This script tests that all dependencies and modules are working correctly.
"""

import sys
import traceback

def test_imports():
    """Test all required imports"""
    print("🧪 Testing Lagos-GAN Environment Setup...")
    print("=" * 50)
    
    tests = [
        ("Core PyTorch", ["torch", "torchvision"]),
        ("Data Science", ["numpy", "pandas", "scipy"]),
        ("Computer Vision", ["cv2", "PIL"]),
        ("Visualization", ["matplotlib.pyplot"]),
        ("Web Apps", ["gradio", "streamlit"]),
        ("Data Processing", ["albumentations", "tqdm"]),
        ("Project Modules", ["afrocover.models", "lagos2duplex.models"]),
        ("Demo App", ["demo.app"]),
    ]
    
    failed_tests = []
    
    for test_name, modules in tests:
        try:
            for module in modules:
                __import__(module)
            print(f"✅ {test_name}: OK")
        except Exception as e:
            print(f"❌ {test_name}: FAILED - {e}")
            failed_tests.append(test_name)
    
    return failed_tests

def test_functionality():
    """Test basic functionality"""
    print("\n🔧 Testing Basic Functionality...")
    print("=" * 50)
    
    try:
        import torch
        import numpy as np
        from demo.app import LagosGANDemo
        
        # Test demo instantiation
        demo = LagosGANDemo()
        print("✅ Demo app creation: OK")
        
        # Test basic tensor operations
        x = torch.randn(2, 3, 256, 256)
        y = torch.mean(x)
        print(f"✅ PyTorch operations: OK (tensor shape: {x.shape})")
        
        # Test numpy operations
        arr = np.random.randn(10, 10)
        mean_val = np.mean(arr)
        print(f"✅ NumPy operations: OK (array shape: {arr.shape})")
        
        return True
        
    except Exception as e:
        print(f"❌ Functionality test failed: {e}")
        return False

def main():
    """Main test function"""
    # Add current directory to path
    sys.path.insert(0, '.')
    
    # Run tests
    failed_imports = test_imports()
    functionality_ok = test_functionality()
    
    print("\n" + "=" * 50)
    if not failed_imports and functionality_ok:
        print("🎉 ALL TESTS PASSED!")
        print("✅ Lagos-GAN environment is ready for use!")
        
        # Print environment info
        import torch
        import numpy as np
        import gradio as gr
        
        print(f"\n📊 Environment Summary:")
        print(f"   • Python: {sys.version.split()[0]}")
        print(f"   • PyTorch: {torch.__version__}")
        print(f"   • NumPy: {np.__version__}")
        print(f"   • Gradio: {gr.__version__}")
        print(f"   • Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
        
        return 0
    else:
        print("❌ SOME TESTS FAILED!")
        if failed_imports:
            print(f"Failed imports: {', '.join(failed_imports)}")
        if not functionality_ok:
            print("Functionality tests failed")
        return 1

if __name__ == "__main__":
    exit(main())
