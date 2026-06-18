import os
import seaborn as sns
import matplotlib.pyplot as plt


def create_eda(train):

    os.makedirs(
        "outputs/figures",
        exist_ok=True
    )

    plt.figure(figsize=(6,4))
    sns.countplot(
        x=train["Transported"]
    )
    plt.title("Target Distribution")
    plt.savefig(
        "outputs/figures/target_distribution.png"
    )
    plt.close()

    plt.figure(figsize=(10,5))
    train.isnull().sum().sort_values(
        ascending=False
    ).plot(kind="bar")

    plt.title("Missing Values")
    plt.tight_layout()
    plt.savefig(
        "outputs/figures/missing_values.png"
    )
    plt.close()

    plt.figure(figsize=(8,5))
    sns.histplot(
        train["Age"],
        bins=30,
        kde=True
    )

    plt.title("Age Distribution")
    plt.savefig(
        "outputs/figures/age_distribution.png"
    )
    plt.close()

    plt.figure(figsize=(8,5))
    sns.countplot(
        data=train,
        x="HomePlanet"
    )

    plt.title("HomePlanet Distribution")
    plt.savefig(
        "outputs/figures/homeplanet_distribution.png"
    )
    plt.close()

    plt.figure(figsize=(8,5))
    sns.countplot(
        data=train,
        x="CryoSleep",
        hue="Transported"
    )

    plt.title(
        "CryoSleep vs Transported"
    )

    plt.savefig(
        "outputs/figures/cryosleep_vs_target.png"
    )

    plt.close()