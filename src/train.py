import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.data_processing import load_data, preprocess_data, split_data, scale_data
from src.config import MODEL_DIR, MODEL_PATH, RANDOM_STATE


def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance
    """
    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    print("\nModel Accuracy:", accuracy)
    print("\nClassification Report:\n", report)

    return accuracy


def save_model(model):
    """
    Save model
    """
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved at: {MODEL_PATH}")


def main():
    print("Loading data...")
    df = load_data()

    print("Preprocessing data...")
    X, y = preprocess_data(df)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(X, y)

    print("Scaling data...")
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    print("Training model...")
    model = train_model(X_train_scaled, y_train)

    print("Evaluating model...")
    evaluate_model(model, X_test_scaled, y_test)

    print("Saving model...")
    save_model(model)

    print("\nTraining pipeline completed successfully!")


if __name__ == "__main__":
    main()
