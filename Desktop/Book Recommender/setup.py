#!/usr/bin/env python3
"""
Setup script for the Semantic Book Recommender.
Installs dependencies and prepares the environment.
"""

import subprocess
import sys
import os

def install_dependencies():
    """Install all required packages."""
    print("📦 Installing dependencies...")

    # Read requirements.txt
    with open('requirements.txt', 'r') as f:
        packages = [line.strip() for line in f if line.strip() and not line.startswith('#')]

    # Install packages
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Failed to install {package}: {e}")
            return False

    print("✅ All dependencies installed!")
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up Semantic Book Recommender")
    print("=" * 50)

    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False

    print(f"✅ Python {sys.version.split()[0]} detected")

    # Install dependencies
    if not install_dependencies():
        return False

    # Verify key imports
    try:
        import sentence_transformers
        import chromadb
        import gradio
        print("✅ Key packages verified")
    except ImportError as e:
        print(f"❌ Import verification failed: {e}")
        return False

    print("\n🎉 Setup complete!")
    print("Run 'python create_embeddings_subset.py all sentence-transformers' to generate embeddings")
    print("Or run 'python main.py' if embeddings are already available")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)