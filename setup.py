#!/usr/bin/env python3
"""
Setup script for the Image Captioning project.

This script sets up the project environment and runs initial tests.
"""

import subprocess
import sys
from pathlib import Path
import os


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible."""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python 3.8+ required, found {version.major}.{version.minor}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install project dependencies."""
    print("📦 Installing dependencies...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        return False
    
    # Install dependencies
    return run_command("pip install -r requirements.txt", "Installing dependencies")


def create_directories():
    """Create necessary directories."""
    print("📁 Creating directories...")
    
    directories = [
        "data/synthetic",
        "models",
        "output",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")
    
    return True


def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    
    # Check if test script exists
    if not Path("test_system.py").exists():
        print("❌ test_system.py not found")
        return False
    
    # Run tests
    return run_command("python test_system.py", "Running system tests")


def main():
    """Main setup function."""
    print("🚀 Image Captioning Project Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("⚠️ Dependency installation failed. You may need to install manually:")
        print("pip install -r requirements.txt")
    
    # Run tests
    if not run_tests():
        print("⚠️ Tests failed. Check the error messages above.")
        print("You can still try running the system manually.")
    
    print("\n🎉 Setup completed!")
    print("\nNext steps:")
    print("1. Start the web interface: streamlit run web_app/app.py")
    print("2. Use the CLI: python cli.py caption <image_path>")
    print("3. Generate sample data: python src/data_generator.py")
    print("4. Read the README.md for more information")


if __name__ == "__main__":
    main()
