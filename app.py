import requests
import streamlit as st


API_URL = "http://127.0.0.1:8000"


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="IMDB Sentiment Classifier",
    page_icon="🎬",
    layout="centered",
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🎬 IMDB Sentiment Classifier")

st.markdown(
    """
    Analyze the sentiment of a movie review using a
    **fine-tuned DistilBERT model**.
    """
)

st.divider()


# --------------------------------------------------
# API status
# --------------------------------------------------

try:
    health_response = requests.get(
        f"{API_URL}/health",
        timeout=3,
    )

    if health_response.status_code == 200:
        st.success("🟢 Model API is online")
    else:
        st.warning("🟡 Model API is responding with an error")

except requests.exceptions.RequestException:
    st.error("🔴 Model API is offline")


# --------------------------------------------------
# Review input
# --------------------------------------------------

st.subheader("Enter a movie review")

review = st.text_area(
    "Review",
    placeholder=(
        "Example: This movie was absolutely fantastic! "
        "I loved every minute of it."
    ),
    height=180,
    label_visibility="collapsed",
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True,
):

    if not review.strip():

        st.warning("Please enter a movie review first.")

    else:

        try:

            with st.spinner("Analyzing review..."):

                response = requests.post(
                    f"{API_URL}/predict",
                    json={"text": review},
                    timeout=30,
                )

            if response.status_code == 200:

                result = response.json()

                sentiment = result["sentiment"]
                confidence = result["confidence"]
                positive_probability = result[
                    "positive_probability"
                ]
                negative_probability = result[
                    "negative_probability"
                ]

                st.divider()

                # ------------------------------------------
                # Main prediction
                # ------------------------------------------

                st.subheader("Prediction")

                if sentiment == "positive":

                    st.success(
                        f"😊 Positive Sentiment\n\n"
                        f"Confidence: {confidence:.2%}"
                    )

                else:

                    st.error(
                        f"😞 Negative Sentiment\n\n"
                        f"Confidence: {confidence:.2%}"
                    )

                # ------------------------------------------
                # Probability metrics
                # ------------------------------------------

                st.subheader("Prediction Probabilities")

                col1, col2 = st.columns(2)

                with col1:

                    st.metric(
                        "😊 Positive",
                        f"{positive_probability:.2%}",
                    )

                with col2:

                    st.metric(
                        "😞 Negative",
                        f"{negative_probability:.2%}",
                    )

                # ------------------------------------------
                # Probability bars
                # ------------------------------------------

                st.write("Positive probability")

                st.progress(
                    positive_probability
                )

                st.write("Negative probability")

                st.progress(
                    negative_probability
                )

            else:

                st.error(
                    f"Prediction failed. "
                    f"API returned status "
                    f"{response.status_code}."
                )

        except requests.exceptions.RequestException:

            st.error(
                "Could not connect to the FastAPI server. "
                "Please make sure the API is running."
            )


# --------------------------------------------------
# About section
# --------------------------------------------------

st.divider()

with st.expander("ℹ️ About this project"):

    st.markdown(
        """
        **IMDB Sentiment Classification**

        This project compares multiple deep learning approaches
        for binary movie-review sentiment classification.

        **Models explored:**

        - Simple RNN
        - LSTM
        - BiLSTM
        - Transformer Encoder
        - DistilBERT

        The final application uses a fine-tuned **DistilBERT**
        model exposed through a **FastAPI inference service**.

        **Architecture:**

        `Streamlit → FastAPI → DistilBERT`
        """
    )

## "python -m streamlit run app.py"