{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "5855f44d-2dab-4f42-8e87-fffb0ea89617",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.ensemble import RandomForestClassifier\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.model_selection import GridSearchCV, cross_val_score\n",
    "from sklearn.metrics import classification_report, confusion_matrix, accuracy_score\n",
    "import pickle\n",
    "import re\n",
    "import nltk\n",
    "from nltk.corpus import stopwords\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b6f7e0dc-e11e-436e-8c3b-e4756f7ad7b4",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "[nltk_data] Downloading package stopwords to\n",
      "[nltk_data]     C:\\Users\\Chavindu_PC\\AppData\\Roaming\\nltk_data...\n",
      "[nltk_data]   Unzipping corpora\\stopwords.zip.\n"
     ]
    }
   ],
   "source": [
    "# Download NLTK data (run once)\n",
    "try:\n",
    "    nltk.data.find('corpora/stopwords')\n",
    "except LookupError:\n",
    "    nltk.download('stopwords')\n",
    "\n",
    "class TextPreprocessor:\n",
    "    \"\"\"Text preprocessing for baseline models\"\"\"\n",
    "    \n",
    "    def __init__(self):\n",
    "        self.stop_words = set(stopwords.words('english'))\n",
    "    \n",
    "    def clean_text(self, text):\n",
    "        \"\"\"Clean and normalize text\"\"\"\n",
    "        if pd.isna(text):\n",
    "            return \"\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "44273f06-df95-4120-8d69-17733e11f32a",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.11"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
