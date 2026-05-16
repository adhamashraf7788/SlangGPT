#!/usr/bin/env python3
"""
Download pre-trained SlangGPT models from Google Drive.
Run: python scripts/download_weights.py
"""

import subprocess
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Folder IDs from Google Drive
# Detection model folder
DETECTION_FOLDER_ID = "1Pfsv4EqK0ySZGEbIC8K1EvdQ5IV1LFHB"
# Generation model folder  
GENERATION_FOLDER_ID = "1WeHz1LPahg-4fFJn2mIaZx4P_F6_KjSO"

# Get project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = PROJECT_ROOT / "model" / "weights"


def check_gdown():
    """Check if gdown is installed."""
    try:
        import gdown
        return True
    except ImportError:
        return False


def install_gdown():
    """Install gdown if not present."""
    print("gdown not found. Installing...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown"])
    print("gdown installed successfully.\n")


def download_folder(folder_id, output_path, description):
    """Download a Google Drive folder."""
    print(f"\n📥 Downloading {description}...")
    print(f"   Folder ID: {folder_id}")
    print(f"   Output: {output_path}")
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build URL
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    
    # Run gdown
    cmd = [
        sys.executable, "-m", "gdown", "--folder",
        url, "-O", str(output_path)
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print(f"✅ {description} downloaded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {description}: {e}")
        return False


def verify_downloads():
    """Verify that model files exist."""
    print("\n" + "="*50)
    print("VERIFYING DOWNLOADS")
    print("="*50)
    
    detection_path = WEIGHTS_DIR / "detection" / "best_model.pt"
    generation_path = WEIGHTS_DIR / "generation" / "best" / "model.safetensors"
    
    all_ok = True
    
    if detection_path.exists():
        size_mb = detection_path.stat().st_size / (1024 * 1024)
        print(f"✅ Detection model: {size_mb:.1f} MB")
    else:
        print(f"❌ Detection model not found at {detection_path}")
        all_ok = False
    
    if generation_path.exists():
        size_gb = generation_path.stat().st_size / (1024 * 1024 * 1024)
        print(f"✅ Generation model: {size_gb:.2f} GB")
    else:
        print(f"❌ Generation model not found at {generation_path}")
        all_ok = False
    
    return all_ok


def main():
    """Main download function."""
    print("="*50)
    print("SlangGPT Model Downloader")
    print("="*50)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Weights will be saved to: {WEIGHTS_DIR}")
    
    # Check/install gdown
    if not check_gdown():
        install_gdown()
    
    # Download detection model
    detection_ok = download_folder(
        DETECTION_FOLDER_ID,
        WEIGHTS_DIR / "detection",
        "Detection Model"
    )
    
    # Download generation model
    generation_ok = download_folder(
        GENERATION_FOLDER_ID,
        WEIGHTS_DIR / "generation",
        "Generation Model"
    )
    
    # Verify downloads
    print("\n")
    if verify_downloads():
        print("\n" + "="*50)
        print("🎉 ALL MODELS DOWNLOADED SUCCESSFULLY!")
        print("="*50)
        print("\nYou can now run:")
        print("  python evaluation/evaluate.py")
        print("  python app/app.py")
    else:
        print("\n⚠️ Some downloads may have failed.")
        print("Try running the script again or download manually from:")
        print(f"  Detection: https://drive.google.com/drive/folders/{DETECTION_FOLDER_ID}")
        print(f"  Generation: https://drive.google.com/drive/folders/{GENERATION_FOLDER_ID}")
        sys.exit(1)


if __name__ == "__main__":
    main()