import pandas as pd

from src.eda import create_eda
from src.preprocess import preprocess
from src.train import train_model
from src.submit import generate_submission

train = pd.read_csv(
    "data/train.csv"
)

test = pd.read_csv(
    "data/test.csv"
)

create_eda(train)

train = preprocess(train)
test = preprocess(test)

drop_cols = [
    "PassengerId",
    "Name",
    "Cabin",
    "Group"
]

features = [
    c for c in train.columns
    if c not in
    drop_cols + ["Transported"]
]

X = train[features]
y = train["Transported"]

X_test = test[features]

model, cat_features = train_model(
    X,
    y
)

generate_submission(
    model,
    X_test,
    test
)