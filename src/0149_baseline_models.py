from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from .0149_tfidf_vectorizer import create_tfidf_vectorizer

def create_logistic_regression_pipeline(tfidf_params=None, model_params=None):
    """
    Create a pipeline with TF-IDF vectorizer and Logistic Regression
    
    Parameters:
    - tfidf_params: dict of parameters for TF-IDF vectorizer
    - model_params: dict of parameters for Logistic Regression
    
    Returns:
    - Configured Pipeline instance
    """
    if tfidf_params is None:
        tfidf_params = {'ngram_range': (1, 1), 'max_features': 10000}
    if model_params is None:
        model_params = {'C': 1.0, 'max_iter': 1000}
    
    pipeline = Pipeline([
        ('tfidf', create_tfidf_vectorizer(**tfidf_params)),
        ('clf', LogisticRegression(**model_params))
    ])
    return pipeline

def create_random_forest_pipeline(tfidf_params=None, model_params=None):
    """
    Create a pipeline with TF-IDF vectorizer and Random Forest
    
    Parameters:
    - tfidf_params: dict of parameters for TF-IDF vectorizer
    - model_params: dict of parameters for Random Forest
    
    Returns:
    - Configured Pipeline instance
    """
    if tfidf_params is None:
        tfidf_params = {'ngram_range': (1, 1), 'max_features': 10000}
    if model_params is None:
        model_params = {'n_estimators': 100, 'max_depth': None}
    
    pipeline = Pipeline([
        ('tfidf', create_tfidf_vectorizer(**tfidf_params)),
        ('clf', RandomForestClassifier(**model_params))
    ])
    return pipeline

def train_and_evaluate(pipeline, X_train, y_train, X_test, y_test):
    """
    Train and evaluate a pipeline
    
    Parameters:
    - pipeline: sklearn Pipeline object
    - X_train, y_train: training data
    - X_test, y_test: test data
    
    Returns:
    - Trained pipeline
    - Classification report
    """
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = pipeline.predict(X_test)
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    
    return pipeline, report

if __name__ == "__main__":
    # Example usage (you'll need to load your actual data)
    from sklearn.datasets import fetch_20newsgroups
    
    # Load example data
    categories = ['alt.atheism', 'soc.religion.christian']
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)
    
    X_train, y_train = newsgroups_train.data, newsgroups_train.target
    X_test, y_test = newsgroups_test.data, newsgroups_test.target
    
    # Create and evaluate Logistic Regression pipeline
    print("Training Logistic Regression...")
    lr_pipeline = create_logistic_regression_pipeline(
        tfidf_params={'ngram_range': (1, 2), 'max_features': 5000},
        model_params={'C': 0.1, 'max_iter': 1000}
    )
    _, lr_report = train_and_evaluate(lr_pipeline, X_train, y_train, X_test, y_test)
    print("Logistic Regression Results:")
    print(lr_report)
    
    # Create and evaluate Random Forest pipeline
    print("\nTraining Random Forest...")
    rf_pipeline = create_random_forest_pipeline(
        tfidf_params={'ngram_range': (1, 2), 'max_features': 5000},
        model_params={'n_estimators': 200, 'max_depth': 50}
    )
    _, rf_report = train_and_evaluate(rf_pipeline, X_train, y_train, X_test, y_test)
    print("Random Forest Results:")
    print(rf_report)