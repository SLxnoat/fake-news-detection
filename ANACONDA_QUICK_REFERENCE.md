# 🐍 Anaconda Quick Reference Card

## 🚀 One-Command Setup

```bash
# Create environment from file (EASIEST METHOD)
conda env create -f environment.yml
conda activate fake-news-detection
```

## 🔑 Essential Commands

### Environment Management
```bash
# Create environment
conda create -n fake-news-detection python=3.9

# Activate environment
conda activate fake-news-detection

# Deactivate environment
conda deactivate

# List environments
conda env list

# Remove environment
conda env remove -n fake-news-detection
```

### Package Installation
```bash
# Install core packages
conda install pandas numpy scikit-learn matplotlib seaborn jupyter

# Install from conda-forge
conda install -c conda-forge streamlit plotly flask flask-cors

# Install from requirements.txt
pip install -r requirements.txt

# List installed packages
conda list
```

### Project Setup
```bash
# Clone repository
git clone <your-repo-url>
cd fake-news-detection

# Install NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

## 📱 Running Applications

### Streamlit App
```bash
# Launch Streamlit
streamlit run app/0169_streamlit_app.py
# Open: http://localhost:8501
```

### API Server
```bash
# Launch Flask API
python app/backend/api_server.py
# API: http://localhost:5000
```

### Jupyter Notebook
```bash
# Launch Jupyter
jupyter notebook
# Open: http://localhost:8888
```

## 🔧 Troubleshooting

### Common Issues
```bash
# Environment not found
conda env list
conda create -n fake-news-detection python=3.9

# Package conflicts
conda update conda
conda clean --all

# Path issues
conda init
# Restart terminal
```

### Verification
```bash
# Check installation
python scripts/check_dependencies.py

# Test imports
python -c "import pandas, numpy, sklearn, nltk, flask, streamlit; print('✅ All packages working!')"
```

## 📁 Project Structure in Anaconda

```
fake-news-detection/
├── 📁 notebooks/          # Jupyter notebooks
├── 📁 app/               # Streamlit & Flask apps
├── 📁 src/               # Source code
├── 📁 data/              # Datasets
├── 📁 models/            # Trained models
├── 📄 environment.yml    # Conda environment file
└── 📄 requirements.txt   # Pip requirements
```

## 🎯 Workflow in Anaconda Navigator

1. **Open Anaconda Navigator**
2. **Go to Environments tab**
3. **Create/Select fake-news-detection environment**
4. **Launch Jupyter Notebook**
5. **Navigate to project folder**
6. **Open notebooks or run applications**

## ⚡ Quick Start Sequence

```bash
# 1. Create environment
conda env create -f environment.yml

# 2. Activate environment
conda activate fake-news-detection

# 3. Clone project
git clone <repo-url>
cd fake-news-detection

# 4. Install NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 5. Run Streamlit app
streamlit run app/0169_streamlit_app.py
```

## 🔍 Package Sources

- **Core ML**: `conda install` (defaults channel)
- **Latest packages**: `conda install -c conda-forge`
- **Special packages**: `pip install`

## 📊 Environment Export/Import

```bash
# Export current environment
conda env export > environment.yml

# Import environment
conda env create -f environment.yml
```

---

**🐍 Anaconda + Fake News Detection = Perfect Match! 🕵️‍♂️**
