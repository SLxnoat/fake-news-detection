#!/usr/bin/env python3
"""
Dependency installation script for Fake News Detection project
This script helps install and verify all required dependencies
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔧 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected. Python 3.8+ is required.")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def upgrade_pip():
    """Upgrade pip to latest version"""
    return run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Upgrading pip"
    )

def install_requirements(requirements_file):
    """Install requirements from a specific file"""
    if not os.path.exists(requirements_file):
        print(f"⚠️ Requirements file {requirements_file} not found, skipping...")
        return False
    
    return run_command(
        f"{sys.executable} -m pip install -r {requirements_file}",
        f"Installing dependencies from {requirements_file}"
    )

def install_nltk_data():
    """Install required NLTK data"""
    print("\n📚 Installing NLTK data...")
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')
        print("✅ NLTK data installed successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to install NLTK data: {e}")
        return False

def verify_installation():
    """Verify that key packages are installed"""
    print("\n🔍 Verifying installation...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'nltk', 'flask', 
        'streamlit', 'plotly', 'matplotlib', 'seaborn'
    ]
    
    all_installed = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            print(f"❌ {package} is NOT available")
            all_installed = False
    
    return all_installed

def main():
    """Main installation function"""
    print("🚀 Fake News Detection - Dependency Installation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Upgrade pip
    if not upgrade_pip():
        print("⚠️ Pip upgrade failed, continuing...")
    
    # Install base requirements
    if not install_requirements("requirements.txt"):
        print("❌ Failed to install base requirements")
        sys.exit(1)
    
    # Install NLTK data
    if not install_nltk_data():
        print("⚠️ NLTK data installation failed")
    
    # Verify installation
    if not verify_installation():
        print("\n❌ Some packages are missing. Please check the errors above.")
        sys.exit(1)
    
    print("\n🎉 All dependencies installed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the dependency checker: python scripts/check_dependencies.py")
    print("2. Test the installation: python scripts/test_installation.py")
    print("3. Start development: jupyter notebook")

if __name__ == "__main__":
    main()
