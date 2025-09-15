#!/usr/bin/env python3

import subprocess
import sys
import os

def install_requirements():
    print("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def setup_directories():
    directories = ['models', 'data', 'data/Training', 'data/Testing']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print("Created directory: " + directory)

def main():
    print("Brain Tumor Detection - Setup Script")
    print("=" * 50)
    
    install_requirements()
    setup_directories()
    
    print("Setup completed successfully!")
    print("Next steps:")
    print("1. Download dataset: cd src && python download_dataset.py")
    print("2. Train model: python train.py")
    print("3. Run web app: streamlit run app.py")

if __name__ == "__main__":
    main()