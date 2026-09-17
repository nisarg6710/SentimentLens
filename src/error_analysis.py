import numpy as np
import pandas as pd


def create_error_dataframe(
    texts,
    true_labels,
    predictions,
    probabilities,
):
    """
    Create a DataFrame containing predictions and error information.
    """

    df = pd.DataFrame({
        "text": texts,
        "true_label": np.asarray(true_labels).astype(int),
        "prediction": np.asarray(predictions).astype(int),
        "probability": np.asarray(probabilities),
    })

    df["correct"] = (
        df["true_label"] == df["prediction"]
    )

    df["confidence"] = np.where(
        df["prediction"] == 1,
        df["probability"],
        1 - df["probability"],
    )

    return df


def get_error_summary(df):
    """
    Return basic error statistics.
    """

    total = len(df)
    incorrect = (~df["correct"]).sum()

    false_positives = (
        (df["true_label"] == 0) &
        (df["prediction"] == 1)
    ).sum()

    false_negatives = (
        (df["true_label"] == 1) &
        (df["prediction"] == 0)
    ).sum()

    return {
        "total_samples": total,
        "correct": int(df["correct"].sum()),
        "incorrect": int(incorrect),
        "false_positives": int(false_positives),
        "false_negatives": int(false_negatives),
        "error_rate": incorrect / total,
    }


def high_confidence_errors(df, threshold=0.90):
    """
    Return errors whose prediction confidence is
    greater than or equal to the given threshold.
    """

    return df[
        (~df["correct"]) &
        (df["confidence"] >= threshold)
    ].sort_values(
        "confidence",
        ascending=False,
    )


def confidence_error_summary(df):
    """
    Count false positives and false negatives
    at different confidence thresholds.
    """

    results = []

    for threshold in [0.90, 0.95, 0.99]:

        confident_errors = high_confidence_errors(
            df,
            threshold,
        )

        false_positives = (
            (confident_errors["true_label"] == 0) &
            (confident_errors["prediction"] == 1)
        ).sum()

        false_negatives = (
            (confident_errors["true_label"] == 1) &
            (confident_errors["prediction"] == 0)
        ).sum()

        results.append({
            "threshold": threshold,
            "false_positives": int(false_positives),
            "false_negatives": int(false_negatives),
            "total": len(confident_errors),
        })

    return pd.DataFrame(results)