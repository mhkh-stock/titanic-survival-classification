# Task 1 — Titanic Survival Classification

**Name:** Muhammad Hashir
**Internship:** Data Science, Arch Technologies (Month 1)
**Dataset:** Kaggle Titanic train.csv — 891 passengers, 12 columns
**Source:** https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv
(This is a public copy of the Kaggle `train.csv`. I checked it after
downloading: 891 rows and 12 columns.)

## What I did

I built a model that guesses if a Titanic passenger survived or not. I used
ticket class, sex, age, fare, family size and port of embarkation. I did the
full work: I looked at the data, cleaned it, made new features, trained two
models and checked how good they are.

## How I did it

1. **I looked at the data first.** 342 passengers survived (38.38%) and 549
   did not. Class 1 survived 62.96%, Class 2 47.28%, Class 3 24.24%.
   Women survived 74.20%, men 18.89%.
2. **I cleaned the data.** I dropped `Cabin` because 687 out of 891 values
   were missing (77%). I filled the 177 missing `Age` values with the median
   age of the passenger's Sex and Ticket Class group. I filled the 2 missing
   `Embarked` values with `S`, the most common port. `Fare` had no missing
   values.
3. **I made new features.** I took the title from the Name (Mr, Mrs, Miss,
   Master, or Rare). I made FamilySize = SibSp + Parch + 1, and an IsAlone
   flag. I changed Sex to 1 for female and 0 for male. I one-hot encoded
   Embarked and Title.
4. **I split the data.** 80% for training (712 passengers) and 20% for
   testing (179 passengers). I kept the same survival rate in both sets
   and I used seed 42 so anyone can get the same result.
5. **I trained two models.** Logistic Regression (with scaled features) and
   Random Forest (300 trees, seed 42).

## My results (test set, 179 passengers)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | **0.8547** | **0.8413** | **0.7681** | **0.8030** | **0.8758** |
| Random Forest | 0.8101 | 0.7612 | 0.7391 | 0.7500 | 0.8266 |

Logistic Regression did better on every score. I learned from this that a
simple model can beat a complex one when the data is small. The most
important features for Random Forest were Fare (0.247), Age (0.216),
Title_Mr (0.127), Sex (0.125) and Pclass (0.076).

## How to run it

```bash
pip install -r requirements.txt
python src/titanic.py
```

## Folder structure

```
task1-titanic/
├── README.md
├── requirements.txt
├── src/
│   └── titanic.py          # my full code, runs end to end
├── data/
│   ├── titanic_raw.csv     # Kaggle train.csv (891 rows)
│   └── titanic_clean.csv   # cleaned data with new features
├── outputs/
│   ├── metrics.json        # all scores and numbers
│   ├── survival_by_class.png
│   ├── survival_by_sex.png
│   ├── age_distribution.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
└── report/
    ├── MuhammadHashir_DataScience_Month1.docx
    └── MuhammadHashir_DataScience_Month1.pdf
```

## Tools I used

Python 3 · pandas · scikit-learn · matplotlib · NumPy
