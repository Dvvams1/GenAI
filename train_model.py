import pandas as pd
import numpy as np
import re
import nltk
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Download stopwords
nltk.download('stopwords')

def train():
    print("Loading dataset...")
    try:
        df = pd.read_csv("Restaurant_Reviews.tsv", delimiter="\t", quoting=3)
    except FileNotFoundError:
        print("Error: Restaurant_Reviews.tsv not found.")
        return

    print("Preprocessing text...")
    corpus = []
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    # Remove 'not' from stopwords to keep negative sentiment
    all_stopwords.remove('not')

    for i in range(0, len(df)):
        # Handle negations and smart quotes
        review = df['Review'][i]
        review = review.replace("’", "'") # Normalize smart quotes
        review = re.sub(r"n't", " not ", review) # Explicit negation
        
        review = re.sub('[^a-zA-Z]', ' ', review)
        review = review.lower()
        review = review.split()
        review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
        review = ' '.join(review)
        corpus.append(review)

    print("Vectorizing...")
    # Using parameters similar to the notebook but ensuring robustness
    cv = TfidfVectorizer(max_features=2500, ngram_range=(1, 2))
    X = cv.fit_transform(corpus).toarray()
    y = df.iloc[:, -1].values

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    print("Training MultinomialNB...")
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc}")

    print("Saving model and vectorizer...")
    joblib.dump(classifier, 'review_model.pkl')
    joblib.dump(cv, 'tfidf.pkl')
    print("Done!")

if __name__ == "__main__":
    train()
