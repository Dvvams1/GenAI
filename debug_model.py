import joblib
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

# Load model
model = joblib.load('review_model.pkl')
tfidf = joblib.load('tfidf.pkl')

# Setup preprocessing
ps = PorterStemmer()
nltk.download('stopwords')
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

def debug_prediction(review):
    print(f"Original Review: {review}")
    
    # Preprocessing steps
    review_clean = re.sub('[^a-zA-Z]', ' ', review)
    print(f"After Regex: {review_clean}")
    
    review_lower = review_clean.lower()
    print(f"After Lower: {review_lower}")
    
    review_split = review_lower.split()
    print(f"After Split: {review_split}")
    
    review_stemmed = [ps.stem(word) for word in review_split if not word in set(all_stopwords)]
    print(f"After Stemming & Stopwords: {review_stemmed}")
    
    review_final = ' '.join(review_stemmed)
    print(f"Final String: {review_final}")
    
    # Vectorization
    vec = tfidf.transform([review_final]).toarray()
    print(f"Vector Non-Zero Count: {vec[0].nonzero()[0].shape[0]}")
    
    # Prediction
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    
    print(f"Prediction: {pred} ({'Positive' if pred == 1 else 'Negative'})")
    print(f"Probabilities: Negative: {prob[0]:.4f}, Positive: {prob[1]:.4f}")

# Test the failing review
debug_prediction("My order was wrong twice, and the staff didn’t seem to care.")

print("\n--- Testing with explicit 'not' ---")
debug_prediction("My order was wrong twice, and the staff did not seem to care.")

