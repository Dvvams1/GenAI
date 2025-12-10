# Project Report: Restaurant Review Classifier

## Problem Statement
The goal of this project is to develop a **Restaurant Review Classifier** that can analyze customer feedback and categorize reviews as **Positive (Liked)** or **Negative (Not Liked)**. This automated solution helps restaurant chains understand customer sentiment efficiently without manual sorting.

## Objectives
- To load and analyze the `Restaurant_Reviews.tsv` dataset.
- To preprocess text data using Natural Language Processing (NLP) techniques.
- To train a **Bernoulli Naive Bayes** model for classification.
- To evaluate the model's accuracy and predict sentiments for new reviews.

## Methodology

### Activity 1: Data Collection & Loading
**Description**: The first step involves importing the necessary libraries and loading the dataset.
- **Libraries Used**: `numpy`, `pandas`, `nltk`, `sklearn`.
- **Dataset**: `Restaurant_Reviews.tsv` (Tab-separated values).
- **Action**: Loaded the dataset using `pd.read_csv()` with `delimiter='\t'`.

### Activity 2: Text Preprocessing
**Description**: Raw text data needs to be cleaned and converted into a format suitable for machine learning.
- **Steps Taken**:
    1.  **Cleaning**: Removed non-alphabetic characters using Regex (`re.sub`).
    2.  **Lowercasing**: Converted all text to lowercase.
    3.  **Stopwords Removal**: Removed common English words (e.g., "the", "is") using `nltk.corpus.stopwords`. *Note: 'not' was preserved to maintain negative sentiment context.*
    4.  **Stemming**: Applied `PorterStemmer` to reduce words to their root form (e.g., "loved" -> "love").
    5.  **Vectorization**: Converted the processed text into a matrix of TF-IDF features using `TfidfVectorizer`.

### Activity 3: Model Training
**Description**: Splitting the data and training the classifier.
- **Data Split**: The dataset was split into **80% Training** and **20% Testing** sets using `train_test_split`.
- **Algorithm**: **Bernoulli Naive Bayes** (`BernoulliNB`) was selected as per the project requirements.
- **Training**: The model was trained on `X_train` and `y_train`.

### Activity 4: Evaluation & Prediction
**Description**: Assessing model performance and testing on new data.
- **Accuracy**: The model achieved an accuracy of approximately **75-77%**.
- **Prediction Logic**: A function `predict_sentiment()` was created to take a new review, preprocess it, and output `'liked'` or `'not liked'`.

## Results
- The classifier successfully categorizes reviews.
- **Example**:
    - Input: "The food was amazing" -> Output: **liked**
    - Input: "Not tasty and the texture was just nasty" -> Output: **not liked**

## Conclusion
The Restaurant Review Classifier provides a robust way to automate sentiment analysis. By using Naive Bayes and NLP techniques, we can effectively gauge customer satisfaction from textual feedback.
