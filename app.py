import streamlit as st

from src.predictor import analyze_news


# Page Config
st.set_page_config(
    page_title="News Category Classifier",
    page_icon="📰",
    layout="centered"
)

# Title
st.title("📰 News Category Classification & Sentiment Analysis")

st.markdown(
    """
    Enter a news headline or article snippet and the model will:
    
    - Predict the News Category
    - Show Prediction Confidence
    - Analyze Sentiment
    - Show Sentiment Confidence
    """
)

# Text Input
news_text = st.text_area(
    "Enter News Headline / Article",
    height=200,
    placeholder="Example: Apple launches a new AI-powered product..."
)

# Analyze Button
if st.button("Analyze News"):

    if news_text.strip() == "":
        st.warning("Please enter some text.")
    else:

        with st.spinner("Analyzing..."):

            result = analyze_news(news_text)

        st.success("Analysis Complete!")

        st.subheader(" Category Prediction")

        st.metric(
            label="Predicted Category",
            value=result["Category"]
        )

        st.metric(
            label="Category Confidence",
            value=f"{result['Category Confidence']}%"
        )

        st.divider()

        st.subheader(" Sentiment Analysis")

        st.metric(
            label="Sentiment",
            value=result["Sentiment"]
        )

        st.metric(
            label="Sentiment Confidence",
            value=f"{result['Sentiment Confidence']}%"
        )

# Footer
st.markdown("---")
st.caption(
    "Built using TF-IDF, Calibrated Linear SVM, Hugging Face Transformers, and Streamlit."
)