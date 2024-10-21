import pandas as pd
from sklearn.model_selection import train_test_split

import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def process_data():

     # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )

    # Séparation des features et de la cible
    X = data.drop(columns='quality')
    y = data['quality']

    # Split des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Sauvegarde des datasets
    X_train.to_csv('./data/X_train.csv', index=False)
    X_test.to_csv('./data/X_test.csv', index=False)
    y_train.to_csv('./data/y_train.csv', index=False)
    y_test.to_csv('./data/y_test.csv', index=False)

if __name__ == "__main__":
    process_data()