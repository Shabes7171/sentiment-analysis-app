from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

class SentimentClassifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('classifier', LogisticRegression(random_state=42))
        ])
        self.is_trained = False

    def train(self, X_train, y_train):
        """Train the sentiment classifier."""
        self.pipeline.fit(X_train, y_train)
        self.is_trained = True

    def predict(self, text):
        """Predict sentiment for given text."""
        if not self.is_trained:
            raise ValueError("Model needs to be trained first!")
        
        return self.pipeline.predict([text])[0]

    def get_prediction_proba(self, text):
        """Get prediction probability."""
        if not self.is_trained:
            raise ValueError("Model needs to be trained first!")
        
        return self.pipeline.predict_proba([text])[0]

    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        if not self.is_trained:
            raise ValueError("Model needs to be trained first!")
        
        y_pred = self.pipeline.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, y_pred),
            'report': classification_report(y_test, y_pred)
        }
