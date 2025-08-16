from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_vectorizer(ngram_range=(1, 1), max_features=None, min_df=1):
    """
    Create a TF-IDF vectorizer with configurable n-grams
    
    Parameters:
    - ngram_range: tuple (min_n, max_n) for n-grams
    - max_features: maximum number of features to keep
    - min_df: minimum document frequency for a term to be kept
    
    Returns:
    - Configured TfidfVectorizer instance
    """
    tfidf_vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        stop_words='english',  # optional: remove English stop words
        sublinear_tf=True     # use sublinear tf scaling
    )
    return tfidf_vectorizer

if __name__ == "__main__":
    # Example usage
    vectorizer = create_tfidf_vectorizer(ngram_range=(1, 2), max_features=5000)
    print("TF-IDF Vectorizer created with:")
    print(f"- N-gram range: {vectorizer.ngram_range}")
    print(f"- Max features: {vectorizer.max_features}")