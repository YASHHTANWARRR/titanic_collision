# Spaceship Titanic - Kaggle Competition

##  Overview

This project is a solution to the Kaggle **Spaceship Titanic** competition. The objective is to predict whether a passenger was transported to an alternate dimension after the Spaceship Titanic collided with a spacetime anomaly.

The project focuses on:

* Exploratory Data Analysis (EDA)
* Missing value handling
* Feature engineering
* Binary classification using CatBoost
* Model evaluation and interpretation
* Kaggle submission generation

---

##  Dataset

The dataset consists of passenger records from the Spaceship Titanic.

### Files

```text
train.csv
test.csv
sample_submission.csv
```

### Target Variable

| Column      | Description                                                             |
| ----------- | ----------------------------------------------------------------------- |
| Transported | Whether the passenger was transported to another dimension (True/False) |

---

##  Exploratory Data Analysis

The notebook generates several visualizations to understand the dataset:

### Dataset Analysis

* Target Distribution
* Missing Values Analysis
* Age Distribution
* HomePlanet Distribution
* Destination Distribution

### Feature Relationships

* CryoSleep vs Transported
* VIP vs Transported
* TotalSpend Distribution
* Deck vs Transported

### Model Evaluation

* Confusion Matrix
* ROC Curve
* Feature Importance

---

## Feature Engineering
Several additional features are extracted from the original dataset.

### Passenger Group Information

Passenger IDs follow the format:

```text
0001_01
```

Features extracted:

* Group
* GroupNumber

### Cabin Information

Cabin values follow the format:

```text
B/45/P
```

Features extracted:

* Deck
* CabinNum
* Side

### Spending Features

A new feature is created:

```python
TotalSpend = (
    RoomService +
    FoodCourt +
    ShoppingMall +
    Spa +
    VRDeck
)
```

Additional feature:

```python
NoSpending
```

---

## Model

The project uses:

### CatBoostClassifier

Reasons for choosing CatBoost:

* Handles categorical features directly
* Requires minimal preprocessing
* Strong performance on tabular datasets
* Excellent handling of missing values

---

##  Evaluation Metrics

The model is evaluated using:

* Accuracy
* Confusion Matrix
* ROC Curve
* Feature Importance Analysis

Typical validation accuracy:

```text
0.80 - 0.83
```

---

##  Project Structure

```text
Spaceship-Titanic/
│
├── train.csv
├── test.csv
├── sample_submission.csv
│
├── spaceship_titanic.ipynb
│
├── outputs/
│   ├── figures/
│   └── submission.csv
│
└── README.md
```

---

##  Installation

Clone the repository:

```bash
git clone <repository-url>
cd Spaceship-Titanic
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn catboost
```

---

## Running the Project

Open the notebook:

```bash
jupyter notebook
```

Run all cells in:

```text
spaceship_titanic.ipynb
```

The notebook will:

1. Load the dataset
2. Perform EDA
3. Create engineered features
4. Train the CatBoost model
5. Evaluate performance
6. Generate predictions
7. Create submission.csv

---

##  Kaggle Submission

The generated submission file follows Kaggle's required format:

```text
PassengerId,Transported
0013_01,False
0018_01,True
0019_01,False
```

Upload the generated:

```text
submission.csv
```

to Kaggle for scoring.

---

##  Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* CatBoost
* Jupyter Notebook

---

##  Results

The final CatBoost model achieves competitive leaderboard performance through:

* Effective handling of missing values
* Cabin feature decomposition
* Passenger grouping information
* Spending-based feature engineering
* Native categorical feature support

---

##  Author

Yash Tanwar
