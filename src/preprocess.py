import pandas as pd
import numpy as np


def preprocess(df):

    df = df.copy()

    df["Group"] = df["PassengerId"].str.split("_").str[0]
    df["GroupNumber"] = df["Group"].astype(int)

    cabin = (
        df["Cabin"]
        .fillna("Unknown/0/U")
        .str.split("/", expand=True)
    )

    df["Deck"] = cabin[0]
    df["CabinNum"] = pd.to_numeric(cabin[1], errors="coerce")
    df["Side"] = cabin[2]

    spend_cols = [
        "RoomService",
        "FoodCourt",
        "ShoppingMall",
        "Spa",
        "VRDeck"
    ]

    df[spend_cols] = df[spend_cols].fillna(0)

    df["TotalSpend"] = df[spend_cols].sum(axis=1)

    df["NoSpending"] = (
        df["TotalSpend"] == 0
    ).astype(int)

    df["Age"] = df["Age"].fillna(
        df["Age"].median()
    )

    df["CryoSleep"] = (
        df["CryoSleep"]
        .fillna(False)
    )

    df["VIP"] = (
        df["VIP"]
        .fillna(False)
    )

    df["HomePlanet"] = (
        df["HomePlanet"]
        .fillna("Unknown")
    )

    df["Destination"] = (
        df["Destination"]
        .fillna("Unknown")
    )

    df["Surname"] = (
        df["Name"]
        .fillna("Unknown Unknown")
        .str.split()
        .str[-1]
    )

    return df