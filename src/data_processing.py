import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.config import DATA_PATH, TEST_SIZE, RANDOM_STATE


def load_data():
    """
    Load dataset from CSV file
    """
    df = pd.read_csv(DATA_PATH)
    return df


def preprocess_data(df):
    """
    Split features and target
    """
    # last column is target
    X = df.drop("target", axis=1)
    y = df["target"]

    return X, y


def split_data(X, y):
    """
    Train-test split
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    """
    Scale numerical features
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler
