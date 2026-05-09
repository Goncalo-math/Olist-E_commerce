from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix,  accuracy_score
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pandas as pd
import numpy as np

FEATURES = [
    "days_to_deliver",
    "delivery_delay",
    "total_payment",
    "n_installments",
    "total_freight",
    "avg_price",
    "n_items",
    "customer_state",
    "payment_type",
]

def train(df):
    df = df[FEATURES + ["is_negative"]].dropna()

    # Encode categoricals
    for col in ["customer_state", "payment_type"]:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    X = df[FEATURES]
    y = df["is_negative"]

    random_seed = 42

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=y
    )

    # Handle class imbalance (negative reviews are a minority)
    sm = SMOTE(random_state=random_seed)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

    model = XGBClassifier()
    model.fit(X_train_res, y_train_res)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")

    ## Confusion Matrix & Cross-validation
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    print("Performing 10-fold cross-validation:")
    accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
    print("Accuracy: {:.2f} %".format(accuracies.mean()*100))
    print("Standard Deviation: {:.2f} %".format(accuracies.std()*100))


    return model