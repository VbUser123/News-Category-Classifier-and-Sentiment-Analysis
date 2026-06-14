import joblib

from src.preprocessing import preprocess_text
from src.sentiment import get_sentiment


model = joblib.load("models/calibrated_svm_news_classifier.pkl")
vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

def analyze_news(text):

    cleaned_text = preprocess_text(text)

    vec = vectorizer.transform([cleaned_text])

    category = model.predict(vec)[0]

    probs = model.predict_proba(vec)[0]

    category_confidence = probs.max()

    sentiment, sentiment_confidence = get_sentiment(text)

    return {
        "Category": category,
        "Category Confidence":
            float(round(category_confidence * 100, 2)),

        "Sentiment": sentiment,

        "Sentiment Confidence":
            round(sentiment_confidence * 100, 2)
    }