# 🐍 Anaconda Navigator Setup Guide for Fake News Detection Project

This guide provides step-by-step instructions for setting up and using the Fake News Detection project through Anaconda Navigator on Windows, macOS, and Linux.

## 🎯 Prerequisites

### System Requirements
- **Anaconda**: Anaconda Distribution or Miniconda
- **Python**: 3.8 or higher (included with Anaconda)
- **RAM**: Minimum 8GB (16GB recommended for BERT models)
- **Storage**: At least 5GB free space
- **OS**: Windows 10/11, macOS 10.15+, or Linux

### Required Software

#### 1. Anaconda Installation
1. **Download Anaconda** from [anaconda.com](https://www.anaconda.com/products/distribution)
2. **Choose the appropriate version** for your operating system
3. **During installation**:
   - ✅ Check "Add Anaconda to PATH" (Windows)
   - ✅ Check "Register Anaconda as default Python"
   - ✅ Check "Install for all users" (recommended)

#### 2. Alternative: Miniconda
If you prefer a minimal installation:
1. **Download Miniconda** from [docs.conda.io](https://docs.conda.io/en/latest/miniconda.html)
2. **Install with default settings**
3. **Install additional packages** as needed

## 🚀 Installation Steps

### Step 1: Launch Anaconda Navigator
1. **Open Anaconda Navigator** from your Start Menu/Applications
2. **Wait for initialization** (first launch may take a few minutes)
3. **Verify installation** by checking the "Environments" tab

### Step 2: Create a New Environment
1. **Click on "Environments"** in the left sidebar
2. **Click "Create"** button at the bottom
3. **Configure the environment**:
   - **Name**: `fake-news-detection`
   - **Python version**: 3.9 or 3.10 (recommended)
   - **Click "Create"**

### Step 3: Clone the Repository
1. **Open Anaconda Prompt** (from Anaconda Navigator or Start Menu)
2. **Navigate to your desired directory**:
   ```bash
   # Windows
   cd C:\Users\%USERNAME%\Documents
   
   # macOS/Linux
   cd ~/Documents
   ```
3. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd fake-news-detection
   ```

### Step 4: Activate the Environment
1. **In Anaconda Prompt**, activate your environment:
   ```bash
   conda activate fake-news-detection
   ```
2. **Verify activation** (you should see `(fake-news-detection)` at the start)

### Step 5: Install Dependencies
1. **Install core packages via conda**:
   ```bash
   conda install pandas numpy scikit-learn matplotlib seaborn jupyter
   conda install -c conda-forge streamlit plotly
   conda install -c conda-forge flask flask-cors
   ```
2. **Install remaining packages via pip**:
   ```bash
   pip install -r requirements.txt
   ```

### Step 6: Install NLTK Data
```bash
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

### Method 1: Environment File (Recommended)
1. **Create environment from file**:
   ```bash
   conda env create -f environment.yml
   ```
2. **Activate the environment**:
   ```bash
   conda activate fake-news-detection
   ```

### Method 2: Automated Installation
```bash
# Run the automated installation script
python scripts/install_dependencies.py
```

### Method 3: Manual Package Installation
```bash
# Install packages one by one
conda install pandas numpy scikit-learn matplotlib seaborn
conda install -c conda-forge streamlit plotly wordcloud
conda install -c conda-forge flask flask-cors
conda install -c conda-forge transformers torch
pip install PyJWT werkzeug
```

## ✅ Verification

### Check Installation
```bash
# Verify conda environment
conda list

# Run the dependency checker
python scripts/check_dependencies.py

# Test Python imports
python -c "import pandas, numpy, sklearn, nltk, flask, streamlit; print('All packages imported successfully!')"
```

### Test the Application
```bash
# Test Streamlit app
streamlit run app/0169_streamlit_app.py

# Test API server (in new terminal)
python app/backend/api_server.py
```

## 🌐 Using Anaconda Navigator

### Launching Applications

#### 1. Jupyter Notebook
1. **In Anaconda Navigator**, go to "Home" tab
2. **Find "Jupyter Notebook"** and click "Launch"
3. **Navigate to your project folder**:
   ```
   fake-news-detection/notebooks/
   ```
4. **Open notebooks**:
   - `0184_EDA_and_Data_Understanding.ipynb`
   - `0149_baseline_training_evaluation.ipynb`
   - `0169_cross_validation.ipynb`

#### 2. JupyterLab (Alternative)
1. **Launch JupyterLab** from Anaconda Navigator
2. **Navigate to your project folder**
3. **Open notebooks** in the file browser

#### 3. Spyder IDE
1. **Launch Spyder** from Anaconda Navigator
2. **Open Python files** from your project
3. **Set working directory** to your project folder

### Managing Environments
1. **View all environments** in the "Environments" tab
2. **Switch between environments** by clicking on them
3. **Install packages** using the search and install interface
4. **Export environment** for sharing with others

## 🚀 Quick Start Commands

### Complete Setup in Anaconda
```bash
# Create and activate environment
conda create -n fake-news-detection python=3.9
conda activate fake-news-detection

# Install packages
conda install pandas numpy scikit-learn matplotlib seaborn jupyter
conda install -c conda-forge streamlit plotly flask flask-cors
pip install -r requirements.txt

# Clone repository (if not done)
git clone <repo-url>
cd fake-news-detection

# Run applications
streamlit run app/0169_streamlit_app.py
```

## 📱 Running Applications

### Streamlit App
```bash
# In Anaconda Prompt with activated environment
streamlit run app/0169_streamlit_app.py

# Open browser to: http://localhost:8501
```

### API Server
```bash
# In new Anaconda Prompt with activated environment
python app/backend/api_server.py

# API available at: http://localhost:5000
```

### Jupyter Notebooks
```bash
# Launch from Anaconda Navigator or command line
jupyter notebook

# Open browser to: http://localhost:8888
```

## 🔧 Troubleshooting

### Common Anaconda Issues

#### 1. Environment Not Found
```bash
# List all environments
conda env list

# Create environment if missing
conda create -n fake-news-detection python=3.9
```

#### 2. Package Installation Failures
```bash
# Update conda
conda update conda

# Try different channels
conda install -c conda-forge package-name

# Use pip for problematic packages
pip install package-name
```

#### 3. Path Issues
```bash
# Check if conda is in PATH
conda --version

# Add Anaconda to PATH manually if needed
# Windows: Add to System Environment Variables
# macOS/Linux: Add to ~/.bashrc or ~/.zshrc
```

#### 4. Permission Errors
```bash
# Use user installation
pip install --user package-name

# Or run Anaconda Prompt as Administrator (Windows)
```

#### 5. Memory Issues
```bash
# Reduce batch sizes in BERT models
# Close other applications
# Use smaller models
```

### Getting Help

1. **Check conda version**: `conda --version`
2. **Check Python version**: `python --version`
3. **List installed packages**: `conda list`
4. **Run dependency checker**: `python scripts/check_dependencies.py`
5. **Check Anaconda Navigator logs**

## 🎯 Best Practices

### Environment Management
1. **Use separate environments** for different projects
2. **Export environment files** for reproducibility
3. **Regularly update conda**: `conda update conda`
4. **Clean unused packages**: `conda clean --all`

### Package Installation
1. **Use conda first** for core scientific packages
2. **Use pip for packages not available in conda**
3. **Specify versions** for reproducibility
4. **Test installations** before proceeding

### Project Organization
1. **Keep project files** in organized folders
2. **Use relative paths** in your code
3. **Document environment setup** for team members
4. **Version control** your environment files

## 🔒 Security Notes

- **Firewall**: Allow Anaconda applications through firewall
- **Antivirus**: Add project directory to exclusions if needed
- **Updates**: Keep Anaconda and packages updated
- **Environment isolation**: Don't install packages in base environment

## 📞 Anaconda-Specific Support

For Anaconda-specific issues:
1. Check Anaconda Navigator logs
2. Verify environment activation
3. Check package compatibility
4. Use conda-forge channel for latest packages
5. Consult [Anaconda documentation](https://docs.anaconda.com/)

## 🚀 Advanced Features

### Using conda-forge
```bash
# Add conda-forge channel
conda config --add channels conda-forge
conda config --set channel_priority strict

# Install packages from conda-forge
conda install -c conda-forge package-name
```

### Environment Export/Import
```bash
# Export environment
conda env export > environment.yml

# Import environment
conda env create -f environment.yml
```

### Package Management
```bash
# Search for packages
conda search package-name

# Install specific version
conda install package-name=version

# Remove packages
conda remove package-name
```

---

**Happy Fake News Detection with Anaconda! 🐍🕵️‍♂️**
