#!/usr/bin/env python3
"""
Test script to verify that all required packages are properly installed
"""

def test_imports():
    """Test all required imports"""
    print("Testing package imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import cv2
        print("✅ OpenCV imported successfully")
    except ImportError as e:
        print(f"❌ OpenCV import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        from ultralytics import YOLO
        print("✅ Ultralytics imported successfully")
    except ImportError as e:
        print(f"❌ Ultralytics import failed: {e}")
        return False
    
    try:
        import torch
        print("✅ PyTorch imported successfully")
        print(f"   PyTorch version: {torch.__version__}")
    except ImportError as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        print("✅ Plotly imported successfully")
    except ImportError as e:
        print(f"❌ Plotly import failed: {e}")
        return False
    
    return True

def test_yolo_model():
    """Test YOLO model loading"""
    print("\nTesting YOLO model loading...")
    
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')
        print("✅ YOLO model loaded successfully")
        return True
    except Exception as e:
        print(f"❌ YOLO model loading failed: {e}")
        print("   This might be due to network issues or insufficient disk space")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("Vehicle Counting & PCU Calculator - Installation Test")
    print("=" * 50)
    
    # Test imports
    imports_ok = test_imports()
    
    if imports_ok:
        print("\n✅ All package imports successful!")
        
        # Test YOLO model
        yolo_ok = test_yolo_model()
        
        if yolo_ok:
            print("\n🎉 Installation test completed successfully!")
            print("You can now run the application with: streamlit run main.py")
        else:
            print("\n⚠️  Package imports successful, but YOLO model loading failed.")
            print("   The application may still work, but the model will be downloaded on first use.")
    else:
        print("\n❌ Some package imports failed.")
        print("   Please check your installation and try again.")
        print("   Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main() 