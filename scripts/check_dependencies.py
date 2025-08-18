#!/usr/bin/env python3
"""
Dependency checker for Fake News Detection project
This script verifies all required packages are installed and accessible
"""

import sys
import importlib
import subprocess
from pathlib import Path

def check_package(package_name, import_name=None):
    """Check if a package is installed and importable"""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        return True, None
    except ImportError as e:
        return False, str(e)

def check_packages():
    """Check all required packages"""
    print("🔍 Checking Required Packages...")
    print("=" * 50)
    
    # Core ML and data science packages
    core_packages = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('scikit-learn', 'sklearn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('nltk', 'nltk'),
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('streamlit', 'streamlit'),
        ('plotly', 'plotly'),
        ('wordcloud', 'wordcloud'),
        ('jupyter', 'jupyter'),
        ('flask', 'flask'),
        ('flask-cors', 'flask_cors'),
        ('scipy', 'scipy'),
        ('pathlib', 'pathlib')
    ]
    
    results = {}
    all_good = True
    
    for package, import_name in core_packages:
        is_installed, error = check_package(package, import_name)
        status = "✅" if is_installed else "❌"
        results[package] = (is_installed, error)
        
        if is_installed:
            print(f"{status} {package}")
        else:
            print(f"{status} {package} - {error}")
            all_good = False
    
    return results, all_good

def check_project_structure():
    """Check if project structure is correct"""
    print("\n📁 Checking Project Structure...")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent
    required_dirs = [
        'data/raw',
        'src/preprocessing',
        'src/models',
        'src/evaluation',
        'src/deployment',
        'app/backend',
        'app/frontend',
        'models',
        'results',
        'notebooks'
    ]
    
    structure_good = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - Missing")
            structure_good = False
    
    return structure_good

def check_data_files():
    """Check if required data files exist"""
    print("\n📊 Checking Data Files...")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent
    data_files = [
        'data/raw/train.tsv',
        'data/raw/test.tsv',
        'data/raw/valid.tsv'
    ]
    
    data_good = True
    
    for file_path in data_files:
        full_path = project_root / file_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"✅ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {file_path} - Missing")
            data_good = False
    
    return data_good

def check_model_files():
    """Check if model files exist"""
    print("\n🤖 Checking Model Files...")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent
    model_files = [
        'models/tfidf_vectorizer.pkl'
    ]
    
    models_good = True
    
    for file_path in model_files:
        full_path = project_root / file_path
        if full_path.exists():
            size_mb = full_path.stat().st_size / (1024 * 1024)
            print(f"✅ {file_path} ({size_mb:.1f} MB)")
        else:
            print(f"⚠️ {file_path} - Missing (will be created during training)")
    
    return True  # Models are optional during setup

def check_python_version():
    """Check Python version compatibility"""
    print("\n🐍 Checking Python Version...")
    print("=" * 50)
    
    version = sys.version_info
    print(f"Current Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible (3.8+)")
        return True
    else:
        print("❌ Python version should be 3.8 or higher")
        return False

def install_missing_packages():
    """Install missing packages using pip"""
    print("\n📦 Installing Missing Packages...")
    print("=" * 50)
    
    missing_packages = [
        'pandas', 'numpy', 'scikit-learn', 'matplotlib', 'seaborn',
        'nltk', 'torch', 'transformers', 'streamlit', 'plotly',
        'wordcloud', 'jupyter', 'flask', 'flask-cors', 'scipy'
    ]
    
    for package in missing_packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")

def main():
    """Main function"""
    print("🚀 Fake News Detection - Dependency Checker")
    print("=" * 60)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check packages
    package_results, packages_ok = check_packages()
    
    # Check project structure
    structure_ok = check_project_structure()
    
    # Check data files
    data_ok = check_data_files()
    
    # Check model files
    models_ok = check_model_files()
    
    # Summary
    print("\n📋 Summary Report:")
    print("=" * 30)
    
    print(f"Python Version: {'✅' if python_ok else '❌'}")
    print(f"Required Packages: {'✅' if packages_ok else '❌'}")
    print(f"Project Structure: {'✅' if structure_ok else '❌'}")
    print(f"Data Files: {'✅' if data_ok else '❌'}")
    print(f"Model Files: {'⚠️' if not models_ok else '✅'}")
    
    # Overall status
    if all([python_ok, packages_ok, structure_ok, data_ok]):
        print("\n🎉 All checks passed! Your environment is ready.")
    else:
        print("\n⚠️ Some checks failed. Here are the issues:")
        
        if not python_ok:
            print("- Python version should be 3.8 or higher")
        
        if not packages_ok:
            print("- Some required packages are missing")
            response = input("\nWould you like to install missing packages? (y/n): ")
            if response.lower() == 'y':
                install_missing_packages()
        
        if not structure_ok:
            print("- Project directory structure is incomplete")
            print("- Please ensure you're running this from the project root")
        
        if not data_ok:
            print("- Required data files are missing")
            print("- Please ensure the LIAR dataset is in data/raw/")
    
    print("\n🔧 Next Steps:")
    if all([python_ok, packages_ok, structure_ok, data_ok]):
        print("1. Run the data processing script: python scripts/create_processed_dataset.py")
        print("2. Test the web application: streamlit run app/backend/app.py")
        print("3. Run notebooks for analysis and training")
    else:
        print("1. Fix the issues identified above")
        print("2. Re-run this checker: python scripts/check_dependencies.py")
        print("3. Then proceed with the next steps")

if __name__ == "__main__":
    main()
