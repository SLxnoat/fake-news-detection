# 🪟 Windows Setup Guide for Fake News Detection Project

This guide provides step-by-step instructions for setting up the Fake News Detection project on Windows 10/11.

## 🎯 Prerequisites

### System Requirements
- **Windows**: 10 (version 1903 or later) or 11
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: At least 5GB free space
- **Python**: 3.8 or higher

### Required Software

#### 1. Python Installation
1. **Download Python** from [python.org](https://www.python.org/downloads/)
2. **Choose Python 3.8+** (latest stable version recommended)
3. **During installation**:
   - ✅ Check "Add Python to PATH"
   - ✅ Check "Install pip"
   - ✅ Check "Install for all users" (recommended)

#### 2. Git Installation
1. **Download Git** from [git-scm.com](https://git-scm.com/download/win)
2. **Install with default settings**
3. **Verify installation**:
   ```cmd
   git --version
   ```

#### 3. Visual Studio Build Tools (Optional)
If you encounter compilation errors:
1. **Download** from [Visual Studio Downloads](https://visualstudio.microsoft.com/downloads/)
2. **Install** "Build Tools for Visual Studio"
3. **Select** "C++ build tools" workload

## 🚀 Installation Steps

### Step 1: Open Command Prompt or PowerShell
- Press `Win + R`, type `cmd` or `powershell`, press Enter
- **Recommended**: Use PowerShell as Administrator

### Step 2: Clone the Repository
```cmd
# Navigate to your desired directory
cd C:\Users\%USERNAME%\Documents

# Clone the repository
git clone <your-repository-url>
cd fake-news-detection
```

### Step 3: Create Virtual Environment
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify activation (you should see (venv) at the start of the line)
```

### Step 4: Install Dependencies
```cmd
# Upgrade pip
python -m pip install --upgrade pip

# Install main requirements
pip install -r requirements.txt

# Install development requirements (optional)
pip install -r requirements-dev.txt
```

### Step 5: Install NLTK Data
```cmd
# Start Python
python

# In Python, run:
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
exit()
```

## 🔧 Alternative Installation Methods

### Method 1: Automated Installation Script
```cmd
# Run the automated installation script
python scripts/install_dependencies.py
```

### Method 2: Using Conda (if you prefer)
```cmd
# Install Miniconda from https://docs.conda.io/en/latest/miniconda.html

# Create conda environment
conda create -n fake-news python=3.9
conda activate fake-news

# Install packages
conda install pandas numpy scikit-learn matplotlib seaborn
pip install -r requirements.txt
```

### Method 3: Minimal Installation (Production)
```cmd
# Install only essential packages
pip install -r requirements-minimal.txt
```

## ✅ Verification

### Check Installation
```cmd
# Run the dependency checker
python scripts/check_dependencies.py

# Test Python imports
python -c "import pandas, numpy, sklearn, nltk, flask, streamlit; print('All packages imported successfully!')"
```

### Test the Application
```cmd
# Test Streamlit app
streamlit run app/0169_streamlit_app.py

# Test API server (in new terminal)
python app/backend/api_server.py
```

## 🐛 Troubleshooting

### Common Windows Issues

#### 1. Python Not Found
```cmd
# Check if Python is in PATH
python --version

# If not found, add Python to PATH manually:
# 1. Search "Environment Variables" in Windows
# 2. Edit "Path" variable
# 3. Add: C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python3x\
# 4. Add: C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python3x\Scripts\
```

#### 2. pip Not Found
```cmd
# Reinstall pip
python -m ensurepip --default-pip
python -m pip install --upgrade pip
```

#### 3. Permission Errors
```cmd
# Run Command Prompt as Administrator
# Or use user installation
pip install --user -r requirements.txt
```

#### 4. Compilation Errors
```cmd
# Install Visual C++ Build Tools
# Or use pre-compiled wheels
pip install --only-binary=all -r requirements.txt
```

#### 5. Virtual Environment Issues
```cmd
# If activation fails, try:
venv\Scripts\activate.bat

# Or recreate the environment
rmdir /s venv
python -m venv venv
venv\Scripts\activate
```

#### 6. Memory Issues
```cmd
# Reduce batch sizes in BERT models
# Close other applications
# Use smaller models
```

### Getting Help

1. **Check Windows version**: `winver`
2. **Check Python version**: `python --version`
3. **Check pip version**: `pip --version`
4. **Check Git version**: `git --version`
5. **Run dependency checker**: `python scripts/check_dependencies.py`

## 🚀 Quick Start Commands

```cmd
# Complete setup in one go
git clone <repo-url>
cd fake-news-detection
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python scripts/install_dependencies.py
streamlit run app/0169_streamlit_app.py
```

## 📱 Running the Applications

### Streamlit App
```cmd
# Start the main application
streamlit run app/0169_streamlit_app.py

# Open browser to: http://localhost:8501
```

### API Server
```cmd
# Start Flask API
python app/backend/api_server.py

# API available at: http://localhost:5000
```

### Jupyter Notebooks
```cmd
# Start Jupyter
jupyter notebook

# Open browser to: http://localhost:8888
```

## 🔒 Security Notes

- **Firewall**: Allow Python and Jupyter through Windows Firewall
- **Antivirus**: Add project directory to antivirus exclusions if needed
- **Updates**: Keep Windows and Python updated

## 📞 Windows-Specific Support

For Windows-specific issues:
1. Check Windows Event Viewer for errors
2. Verify Windows Defender settings
3. Check for Windows updates
4. Ensure proper user permissions

---

**Happy Fake News Detection on Windows! 🪟🕵️‍♂️**
