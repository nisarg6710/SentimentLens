# Error Analysis

## DistilBERT

The final DistilBERT model was evaluated on the untouched
25,000-review IMDB test set.

- Accuracy: 92.97%
- Incorrect predictions: 1,758
- False positives: 999
- False negatives: 759
- Brier Score: 0.0586

### Confidence Analysis

Among the incorrect predictions:

| Confidence threshold | False Positives | False Negatives | Total |
|---|---:|---:|---:|
| >= 90% | 622 | 430 | 1052 |
| >= 95% | 517 | 328 | 845 |
| >= 99% | 261 | 86 | 347 |

A substantial fraction of errors were made with high confidence.
This indicates that the model can be confidently wrong on difficult
or ambiguous reviews.

### Review Length

Reviews longer than 500 tokens had:

- Error rate: 9.87%

Reviews with 500 tokens or fewer had:

- Error rate: 6.54%

The longer-review group therefore had a higher observed error rate.

This does not prove that truncation caused the errors. Longer reviews
may also contain more complex or mixed sentiment.

### Qualitative Error Patterns

Inspection of high-confidence errors showed several recurring patterns:

1. Mixed sentiment
2. Contradictory positive and negative statements
3. Strong local sentiment cues that did not represent the overall review
4. Ambiguous or unusual review text
5. Reviews containing unusual/repetitive wording

### Conclusion

DistilBERT substantially outperformed the from-scratch models, but
remaining errors are concentrated in reviews where sentiment is
ambiguous, mixed, or difficult to infer from the text.

High-confidence errors are particularly important because they show
that confidence alone does not guarantee correctness.


---

## Scratch Transformer

The scratch Transformer achieved:

- Accuracy: 84.58%
- Precision: 84.70%
- Recall: 84.41%
- F1: 84.55%
- ROC-AUC: 92.61%
- Brier Score: 0.1094

The model made 3,855 errors on the 25,000-review test set.

- False positives: 1,906
- False negatives: 1,949

The observed error rate was:

- <=500 tokens: 15.18%
- >500 tokens: 18.40%

The same general pattern was observed: longer reviews had a higher
error rate.

Qualitative inspection showed errors involving mixed sentiment,
contradictory cues, unusual reviews, and cases where strong local
sentiment signals conflicted with the overall review label.