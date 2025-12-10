from flask import Flask, render_template, request, jsonify
import joblib
import re
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk

app = Flask(__name__)

# Load model and vectorizer
try:
    model = joblib.load('review_model.pkl')
    tfidf = joblib.load('tfidf.pkl')
except:
    print("Model files not found. Please run train_model.py first.")
    model = None
    tfidf = None

# Initialize PorterStemmer and Stopwords
ps = PorterStemmer()
nltk.download('stopwords')
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')

def preprocess_review(review):
    # Handle negations and smart quotes
    review = review.replace("’", "'") # Normalize smart quotes
    review = re.sub(r"n't", " not ", review) # Explicit negation
    
    review = re.sub('[^a-zA-Z]', ' ', review)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(all_stopwords)]
    review = ' '.join(review)
    return review

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not tfidf:
        return jsonify({'error': 'Model not loaded'}), 500

    data = request.get_json()
    review = data.get('review', '')
    
    if not review:
        return jsonify({'error': 'No review provided'}), 400

    processed_review = preprocess_review(review)
    vectorized_review = tfidf.transform([processed_review]).toarray()
    prediction = model.predict(vectorized_review)[0]
    
    result = "Positive" if prediction == 1 else "Negative"
    sentiment = "😊" if prediction == 1 else "😞"
    
    return jsonify({'result': result, 'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)
