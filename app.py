import streamlit as st
import pandas as pd

from src.inference import SentimentInference


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SentimentLens",
    page_icon="🔎",
    layout="wide",
)


# --------------------------------------------------
# Styling
# --------------------------------------------------

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 2rem;
        border-radius: 18px;
        margin-bottom: 1.5rem;
        background: linear-gradient(
            135deg,
            #111827,
            #374151
        );
    }

    .hero h1 {
        font-size: 2.7rem;
        margin-bottom: 0.3rem;
    }

    .hero p {
        font-size: 1.1rem;
        opacity: 0.8;
    }

    .priority-high {
        font-size: 1.2rem;
        font-weight: 700;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 650;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return SentimentInference()


try:
    model = load_model()
except Exception as error:
    st.error("Unable to load the sentiment model.")
    st.exception(error)
    st.stop()


# --------------------------------------------------
# Session state
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if "batch_results" not in st.session_state:
    st.session_state.batch_results = None


# --------------------------------------------------
# Header
# --------------------------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🔎 SentimentLens</h1>
        <p>
            Customer Voice Intelligence Platform
        </p>
        <p style="font-size: 1rem; opacity: 0.7;">
            Turn customer feedback into actionable sentiment insights.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "AI-powered sentiment analysis demonstrated using the IMDB benchmark dataset."
)


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("Navigation")

    page = st.radio(
        "Go to",
        [
            "📊 Dashboard",
            "🔍 Analyze Feedback",
            "📁 Batch Analysis",
            "🧠 Model Insights",
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.subheader("Model")

    st.write("**DistilBERT**")
    st.write("Binary sentiment classification")

    st.divider()

    st.caption(
        "Fine-tuned Transformer model for customer feedback "
        "sentiment classification."
    )


# ==================================================
# DASHBOARD
# ==================================================

if page == "📊 Dashboard":

    st.markdown(
        '<div class="section-title">Customer Voice Overview</div>',
        unsafe_allow_html=True,
    )

    history = st.session_state.history

    if history:

        total = len(history)

        positive = sum(
            item["sentiment"] == "positive"
            for item in history
        )

        negative = total - positive

        positive_rate = positive / total
        negative_rate = negative / total

        avg_confidence = sum(
            item["confidence"]
            for item in history
        ) / total

    else:

        total = 0
        positive = 0
        negative = 0
        positive_rate = 0
        negative_rate = 0
        avg_confidence = 0


    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Reviews Analyzed",
            total,
        )

    with col2:
        st.metric(
            "Positive",
            f"{positive_rate * 100:.1f}%",
        )

    with col3:
        st.metric(
            "Negative",
            f"{negative_rate * 100:.1f}%",
        )

    with col4:
        st.metric(
            "Avg. Confidence",
            f"{avg_confidence * 100:.1f}%",
        )


    st.divider()

    if history:

        left, right = st.columns(2)

        with left:

            st.subheader("Sentiment Distribution")

            chart_data = pd.DataFrame(
                {
                    "Sentiment": [
                        "Positive",
                        "Negative",
                    ],
                    "Reviews": [
                        positive,
                        negative,
                    ],
                }
            )

            st.bar_chart(
                chart_data.set_index("Sentiment")
            )

        with right:

            st.subheader("Recent Predictions")

            history_df = pd.DataFrame(history)

            history_df["sentiment"] = (
                history_df["sentiment"]
                .str.capitalize()
            )

            history_df["confidence"] = (
                history_df["confidence"] * 100
            ).round(2).astype(str) + "%"

            history_df.columns = [
                "Sentiment",
                "Confidence",
            ]

            st.dataframe(
                history_df,
                use_container_width=True,
                hide_index=True,
            )

    else:

        st.info(
            "No reviews analyzed yet. "
            "Go to 'Analyze Feedback' to start."
        )


# ==================================================
# SINGLE REVIEW
# ==================================================

elif page == "🔍 Analyze Feedback":

    st.subheader("Analyze Customer Feedback")

    examples = {
        "Positive example": (
            "An absolutely fantastic movie with brilliant "
            "performances and a story that kept me interested."
        ),
        "Negative example": (
            "This movie was painfully boring. "
            "The story was terrible and the acting was awful."
        ),
        "Mixed example": (
            "The acting was excellent and the visuals "
            "were impressive, but the story was predictable."
        ),
    }

    example = st.selectbox(
        "Try an example",
        ["None"] + list(examples.keys()),
    )

    if example != "None":
        review_default = examples[example]
    else:
        review_default = ""


    review = st.text_area(
        "Customer feedback",
        value=review_default,
        height=220,
        placeholder="Enter customer feedback here...",
    )


    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Characters",
            len(review),
        )

    with col2:
        st.metric(
            "Words",
            len(review.split()),
        )

    with col3:
        st.metric(
            "Estimated Tokens",
            int(len(review.split()) * 1.3)
            if review
            else 0,
        )


    analyze = st.button(
        "✨ Analyze Feedback",
        type="primary",
        use_container_width=True,
    )


    if analyze:

        if not review.strip():

            st.warning(
                "Please enter some feedback first."
            )

        else:

            with st.spinner(
                "Analyzing customer feedback..."
            ):

                result = model.predict(review)


            sentiment = result["sentiment"]
            confidence = result["confidence"]
            positive_probability = result[
                "positive_probability"
            ]
            negative_probability = result[
                "negative_probability"
            ]


            # Determine business priority

            if sentiment == "negative":

                if confidence >= 0.90:
                    priority = "🔴 High"
                elif confidence >= 0.70:
                    priority = "🟠 Medium"
                else:
                    priority = "🟡 Review"

            else:

                priority = "🟢 Low"


            # Save result

            st.session_state.history.insert(
                0,
                {
                    "sentiment": sentiment,
                    "confidence": confidence,
                },
            )

            st.session_state.history = (
                st.session_state.history[:20]
            )


            st.divider()

            st.subheader("Prediction")

            result_col1, result_col2, result_col3 = (
                st.columns(3)
            )

            with result_col1:

                if sentiment == "positive":
                    st.success("😊 POSITIVE")
                else:
                    st.error("😞 NEGATIVE")

            with result_col2:

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%",
                )

            with result_col3:

                st.metric(
                    "Feedback Priority",
                    priority,
                )


            st.subheader(
                "Sentiment Probability"
            )

            positive_col, negative_col = st.columns(2)

            with positive_col:

                st.write("😊 Positive")

                st.progress(
                    positive_probability,
                    text=(
                        f"{positive_probability * 100:.2f}%"
                    ),
                )

            with negative_col:

                st.write("😞 Negative")

                st.progress(
                    negative_probability,
                    text=(
                        f"{negative_probability * 100:.2f}%"
                    ),
                )


# ==================================================
# BATCH ANALYSIS
# ==================================================

elif page == "📁 Batch Analysis":

    st.subheader("Batch Feedback Analysis")

    st.write(
        "Upload a CSV containing a column named "
        "`text` containing customer reviews."
    )

    uploaded_file = st.file_uploader(
        "Upload feedback CSV",
        type=["csv"],
    )


    if uploaded_file:

        try:

            data = pd.read_csv(uploaded_file)

        except Exception as error:

            st.error("Could not read the CSV file.")
            st.exception(error)
            st.stop()


        if "text" not in data.columns:

            st.error(
                "The CSV must contain a column named `text`."
            )

        else:

            st.success(
                f"{len(data):,} reviews loaded."
            )


            if st.button(
                "🚀 Analyze All Reviews",
                type="primary",
                use_container_width=True,
            ):

                progress = st.progress(0)

                results = []

                total_rows = len(data)

                for index, text in enumerate(
                    data["text"]
                ):

                    if not isinstance(text, str):
                        text = str(text)

                    prediction = model.predict(text)

                    sentiment = prediction["sentiment"]
                    confidence = prediction["confidence"]

                    if sentiment == "negative":

                        if confidence >= 0.90:
                            priority = "High"
                        elif confidence >= 0.70:
                            priority = "Medium"
                        else:
                            priority = "Review"

                    else:

                        priority = "Low"


                    results.append(
                        {
                            "text": text,
                            "sentiment": sentiment,
                            "confidence": confidence,
                            "priority": priority,
                        }
                    )

                    progress.progress(
                        (index + 1) / total_rows
                    )


                st.session_state.batch_results = (
                    pd.DataFrame(results)
                )

                st.success(
                    "Batch analysis completed."
                )


    # Display results

    if st.session_state.batch_results is not None:

        results = st.session_state.batch_results

        total = len(results)

        positive = (
            results["sentiment"] == "positive"
        ).sum()

        negative = (
            results["sentiment"] == "negative"
        ).sum()

        avg_confidence = (
            results["confidence"].mean()
        )


        st.divider()

        st.subheader("Business Overview")


        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Reviews",
                f"{total:,}",
            )

        with col2:
            st.metric(
                "Positive",
                f"{positive:,}",
            )

        with col3:
            st.metric(
                "Negative",
                f"{negative:,}",
            )

        with col4:
            st.metric(
                "Avg. Confidence",
                f"{avg_confidence * 100:.1f}%",
            )


        st.subheader("Sentiment Distribution")

        distribution = pd.DataFrame(
            {
                "Sentiment": [
                    "Positive",
                    "Negative",
                ],
                "Reviews": [
                    positive,
                    negative,
                ],
            }
        )

        st.bar_chart(
            distribution.set_index("Sentiment")
        )


        st.subheader("Review Explorer")

        sentiment_filter = st.selectbox(
            "Filter sentiment",
            [
                "All",
                "Positive",
                "Negative",
            ],
        )


        filtered = results.copy()

        if sentiment_filter != "All":

            filtered = filtered[
                filtered["sentiment"]
                == sentiment_filter.lower()
            ]


        display_df = filtered.copy()

        display_df["confidence"] = (
            display_df["confidence"] * 100
        ).round(2).astype(str) + "%"

        display_df["sentiment"] = (
            display_df["sentiment"].str.capitalize()
        )


        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
        )


        # Download results

        csv = results.to_csv(
            index=False
        ).encode("utf-8")


        st.download_button(
            "⬇️ Download Analysis",
            data=csv,
            file_name="sentiment_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )


# ==================================================
# MODEL INSIGHTS
# ==================================================

elif page == "🧠 Model Insights":

    st.subheader("Model Performance")

    comparison = pd.DataFrame(
        {
            "Model": [
                "Simple RNN",
                "LSTM",
                "BiLSTM",
                "Transformer",
                "DistilBERT",
            ],
            "Accuracy": [
                75.45,
                83.73,
                85.42,
                84.58,
                92.97,
            ],
            "F1 Score": [
                74.87,
                84.18,
                85.22,
                84.55,
                93.03,
            ],
            "ROC-AUC": [
                82.02,
                91.24,
                93.24,
                92.61,
                97.98,
            ],
        }
    )


    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True,
    )


    st.subheader("Accuracy Comparison")

    st.bar_chart(
        comparison.set_index("Model")["Accuracy"]
    )


    st.divider()

    st.subheader("Production Model")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Accuracy",
            "92.97%",
        )

    with col2:
        st.metric(
            "F1 Score",
            "93.03%",
        )

    with col3:
        st.metric(
            "ROC-AUC",
            "97.98%",
        )


    with st.expander(
        "⚙️ System Architecture"
    ):

        st.markdown(
            """
            **Customer Feedback**

            ↓

            **Streamlit Interface**

            ↓

            **DistilBERT Tokenizer**

            ↓

            **Fine-tuned DistilBERT**

            ↓

            **Sentiment Probabilities**

            ↓

            **Business Feedback Priority**
            """
        )


    with st.expander(
        "🔬 Model Development"
    ):

        st.write(
            """
            The project evaluated five approaches:

            • Simple RNN  
            • LSTM  
            • BiLSTM  
            • Transformer Encoder  
            • Fine-tuned DistilBERT

            The same IMDB train/validation/test framework
            was used for model evaluation.
            """
        )


    with st.expander(
        "🚀 Engineering Stack"
    ):

        st.write(
            """
            Python  
            PyTorch  
            Hugging Face Transformers  
            FastAPI  
            Streamlit  
            Docker  
            GitHub Actions  
            GitHub Container Registry  
            """
        )