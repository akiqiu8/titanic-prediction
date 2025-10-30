import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("=== Loading and preparing training data ===")

    # Load train.csv
    df = pd.read_csv('data/train.csv')
    print(f"Train shape before cleaning: {df.shape}")

    # Clean data
    df.drop(columns=['Cabin'], inplace=True)
    df.dropna(inplace=True)
    print(f"Train shape after dropping NAs: {df.shape}")
    df['LogFare'] = np.log(df['Fare'] + 1)

    # Split features/target
    numeric_features = ['Age', 'LogFare', 'SibSp', 'Parch']
    categorical_features = ['Pclass', 'Sex', 'Embarked']
    X = df[numeric_features + categorical_features]
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=400, stratify=y
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    print("\n=== Setting up preprocessing and model pipeline ===")

    # Preprocessing pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(drop='first', handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    # Logistic regression model inside a pipeline
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000))
    ])

    # Hyperparameter tuning
    param_grid = {
        'classifier__C': [0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l2']
    }

    print("\n=== Running GridSearchCV to tune hyperparameter C ===")
    grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    print("Best parameters found:", grid_search.best_params_)

    # Evaluate on test split
    y_pred = best_model.predict(X_test)
    print("\n=== Model Evaluation on test set of train.csv ===")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))



    # Load test.csv & predict
    print("\n=== Predicting on the actual Titanic test.csv ===")
    test = pd.read_csv("data/test.csv")
    print(f"Loaded test.csv with shape: {test.shape}")

    test.drop(columns=['Cabin'], inplace=True, errors='ignore')
    test.ffill(inplace=True)

    test['LogFare'] = np.log(test['Fare'] + 1)

    X_test_final = test[numeric_features + categorical_features]
    test_preds = best_model.predict(X_test_final)
    pred_df = pd.DataFrame({
        'PassengerId': test['PassengerId'],
        'Survived': test_preds
    }).set_index('PassengerId')
    pred_df.to_csv("data/python_test_predictions.csv")
    print("Predictions saved to 'src/data/python_test_predictions.csv'")


if __name__ == "__main__":
    main()