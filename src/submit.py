import pandas as pd


def generate_submission(
    model,
    X_test,
    test
):

    pred = model.predict(X_test)

    submission = pd.DataFrame({
        "PassengerId":
        test["PassengerId"],

        "Transported":
        pred.astype(bool)
    })

    submission.to_csv(
        "outputs/submission.csv",
        index=False
    )

    print(
        "Submission file saved!"
    )