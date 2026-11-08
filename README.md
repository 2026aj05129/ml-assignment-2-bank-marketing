# ML Assignment 2: Bank Term-Deposit Subscription Classifier

## a. Problem Statement

Portuguese bank telemarketing campaigns call existing clients to sell term
deposits, but only a small fraction of calls convert. The goal of this project
is to predict whether a client will subscribe to a term deposit (`y = yes/no`)
using demographic, financial, and campaign attributes. Five classifiers are
trained on the same split and compared using six evaluation metrics.

## b. Dataset Description

- **Name:** Bank Marketing (`bank-full.csv`)
- **Source:** [UCI Machine Learning Repository: Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- **Size:** 45,211 instances
- **Features:** 16 input attributes: client data (`age`,
  `job`, `marital`, `education`, `default`, `balance`, `housing`, `loan`),
  contact data (`contact`, `day`, `month`), and campaign data (`duration`,
  `campaign`, `pdays`, `previous`, `poutcome`)
- **Target:** `y`: did the client subscribe to a term deposit? (binary:
  `yes` / `no`)
- **Class balance:** highly imbalanced, about 88.3% `no` vs 11.7% `yes`,
  which is why metrics beyond accuracy (AUC, F1, MCC) matter here.
- **Leakage handling:** the `duration` column is **excluded** from training.
  Call duration is only known *after* a call finishes, so using it would leak
  information unavailable at prediction time (as the UCI documentation itself
  warns). That leaves 15 predictors. Other campaign attributes must also be
  checked for availability at the intended prediction time; this experiment
  does not establish that every feature is available before a call.

## c. GitHub Repository

**GitHub repository:** https://github.com/2026aj05129/ml-assignment-2-bank-marketing

**Live Streamlit app:** https://2026aj05129-bank-marketing.streamlit.app/

## d. Models Used and Comparison

All five classifiers share the same preprocessing (median imputation +
standardization for numeric columns; most-frequent imputation + one-hot
encoding for categoricals) inside a single scikit-learn `Pipeline`, trained on
the same stratified 80/20 split (`random_state=42`): 36,168 training rows
and 9,043 held-out test rows. The test set contains 7,985 `no` and 1,058 `yes`
labels and is saved as `test_data.csv`. Precision, recall and F1 use `yes` as
the positive class. AUC uses predicted probabilities, or decision scores for
the SVM. The test set is also used to choose the reported winner, so these
results are not an independent estimate of the selected model's performance.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8933 | 0.7717 | 0.6632 | 0.1786 | 0.2815 | 0.3065 |
| Decision Tree | 0.8308 | 0.6094 | 0.2948 | 0.3204 | 0.3071 | 0.2111 |
| K-Nearest Neighbors | 0.8920 | 0.7266 | 0.6311 | 0.1843 | 0.2853 | 0.3008 |
| Gaussian Naive Bayes | 0.8452 | 0.7514 | 0.3708 | 0.4641 | 0.4123 | 0.3271 |
| Support Vector Machine | 0.8932 | 0.6982 | 0.6503 | 0.1881 | 0.2918 | 0.3105 |

> Note: a majority-class guesser already scores ~88.3% accuracy on this
> dataset, so accuracy alone is a poor yardstick; AUC, F1, and MCC drive the
> conclusions below.

### Observations

| Model | Observation |
|---|---|
| Logistic Regression | Highest AUC (0.772) and precision (0.663). It correctly identifies 189 of 1,058 subscribers (recall 0.179) and produces 96 false positives. |
| Decision Tree | Lowest AUC (0.609), accuracy (0.831) and MCC (0.211). It identifies 339 subscribers but produces 811 false positives. Training performance was not evaluated here, so overfitting cannot be concluded from this table alone. |
| K-Nearest Neighbors | Accuracy (0.892) and recall (0.184) are close to Logistic Regression, but AUC is lower (0.727). It identifies 195 subscribers and produces 114 false positives. |
| Gaussian Naive Bayes | Highest recall (0.464), F1 (0.412) and MCC (0.327). It identifies 491 subscribers, but produces 833 false positives and has precision 0.371. |
| Support Vector Machine | Accuracy (0.893) is close to Logistic Regression, with recall 0.188 and AUC 0.698. It identifies 199 subscribers and produces 107 false positives. |
| **Overall winner** | **Gaussian Naive Bayes**, if the goal is to prioritize F1 and MCC at the default prediction threshold. Logistic Regression is the AUC and precision leader. The preferred model depends on the costs of missed subscribers versus extra calls; no threshold tuning or cost analysis was performed. |

## How to run locally

Use **Python 3.12** and the pinned dependencies in `requirements.txt` to
ensure compatibility with the saved models:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Upload the provided `test_data.csv` (also available from the app) and select a
model to see its metrics, confusion matrix, and classification report. Other
test CSVs must contain the 15 model inputs and the `y` target with both `no`
and `yes` labels. The app loads saved models; it does not retrain them.

To retrain the models, download `bank-full.csv` from the UCI source above and
place it in `data/`, then run `python train_models.py`.
