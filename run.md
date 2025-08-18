## Run Guide

### Quick start (Windows PowerShell)
- **Create/activate env**
```powershell
# Using conda (recommended)
conda create -n fake-news-detection python=3.10 -y
conda activate fake-news-detection

# Or using venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

- **Install dependencies**
```powershell
pip install -r requirements.txt
```

- **Configure environment (optional but recommended)**
```powershell
# Copy example and edit with your values
Copy-Item env.example .env
# Edit .env with your secret keys and passwords
```

- **Train baseline models (optional, recommended)**
```powershell
python scripts\train_baseline_models.py
```

- **Run the Streamlit app**
```powershell
python -m streamlit run app\unified_app.py
```

- **Run the API server**
```powershell
python app\backend\api_server.py
```

### Access
- **App UI**: http://localhost:8501
- **API health**: http://localhost:5000/api/health

### Minimal workflow
1. Install deps: `pip install -r requirements.txt`
2. (Optional) Train: `python scripts\train_baseline_models.py`
3. Start UI: `python -m streamlit run app\unified_app.py`

### API quick test
```powershell
# Health
Invoke-WebRequest http://localhost:5000/api/health | Select-Object -ExpandProperty Content
```

### Security notes
- **Production**: Set `SECRET_KEY`, `ADMIN_PASSWORD`, `API_USER_PASSWORD` in environment variables
- **Development**: Default values are used if `.env` not found (insecure for production)
- **API Auth**: Use JWT tokens from `/api/login` endpoint for protected routes

### Notes
- The app includes a lightweight fallback model if trained models are missing (`models/baseline/*.pkl`). Training improves results.
- NLTK data (stopwords, punkt, punkt_tab) downloads automatically at first run.
- **API now uses real model predictions** when trained models are available, falls back to simulation only when no models loaded.

### Troubleshooting
- **Python not found on Windows**: Disable App Execution Aliases for Python, or use Anaconda Prompt, or ensure PATH is set.
- **NLTK resource error**: If prompted, run once:
```powershell
python - <<'PY'
import nltk
for p in ['stopwords','punkt','punkt_tab']:
    try:
        nltk.download(p)
    except Exception:
        pass
print('Done')
PY
```
- **Port in use**: Change Streamlit port `--server.port 8502` or stop other processes using 8501/5000.
- **Missing models**: Run `python scripts\train_baseline_models.py`.
- **CORS/JWT errors on API**: Ensure `flask-cors` and `PyJWT` are installed (already in `requirements.txt`).
- **API predictions seem random**: Train models first with `python scripts\train_baseline_models.py` for real predictions.
