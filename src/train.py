from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def train_model(X, y):

    cat_features = []

    for c in X.columns:
        if X[c].dtype == "object":
            cat_features.append(c)

    X_train, X_val, y_train, y_val = (
        train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )
    )

    model = CatBoostClassifier(
        iterations=1500,
        depth=8,
        learning_rate=0.03,
        verbose=200,
        random_seed=42
    )

    model.fit(
        X_train,
        y_train,
        cat_features=cat_features,
        eval_set=(X_val, y_val)
    )

    pred = model.predict(X_val)

    score = accuracy_score(
        y_val,
        pred
    )

    print(
        f"Validation Accuracy: {score:.4f}"
    )

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance":
        model.get_feature_importance()
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    plt.figure(figsize=(12,8))

    sns.barplot(
        data=importance.head(20),
        x="Importance",
        y="Feature"
    )

    plt.tight_layout()

    plt.savefig(
        "outputs/figures/feature_importance.png"
    )

    plt.close()

    return model, cat_features