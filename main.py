# Heart Disease Prediction using Machine Learning

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


# 1 Load Dataset
df = pd.read_csv("heart_disease.csv")  # ensure it's in same folder
print("First 5 rows:")
print(df.head())
print("\nColumns:", df.columns.tolist())

# 2 Basic Info & Missing Values
print("\nInfo:")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())


# 3 Correlation Heatmap (to see relationships)
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()


# 4 Features (X) and Target (y)
X = df[["age", "sex", "resting_bp", "cholesterol",
        "max_heart_rate", "exercise_angina"]]
y = df["heart_disease"]  # 1 = disease, 0 = no disease


# 5 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")


# 6 Feature Scaling
# Many ML models work better if features are on similar scale.
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 7 Model - Random Forest Classifier
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
model.fit(X_train_scaled, y_train)

print("\nModel training completed!")


# 8 Model Evaluation
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n Accuracy:", acc)
print("\n Confusion Matrix:\n", cm)
print("\n Classification Report:\n", classification_report(y_test, y_pred))


# 9 Plot Confusion Matrix
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()


# 10 Predict for a New Patient
# Change these values to test different cases
new_patient = pd.DataFrame({
    "age": [52],
    "sex": [1],               # 1 = male, 0 = female
    "resting_bp": [150],
    "cholesterol": [260],
    "max_heart_rate": [140],
    "exercise_angina": [1]    # 1 = yes, 0 = no
})

new_patient_scaled = scaler.transform(new_patient)
new_pred = model.predict(new_patient_scaled)[0]

print("\n❤ Prediction for new patient (1 = disease, 0 = no disease):", new_pred)
