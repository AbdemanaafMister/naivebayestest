# Algorithm: Gaussian Naive Bayes

# 1. IMPORT LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    RocCurveDisplay
)

RANDOM_STATE = 42
sns.set_theme(style="whitegrid")


# 2. LOAD THE DATASET
dataset = load_breast_cancer(as_frame=True)

X = dataset.data
y = dataset.target

class_names = dataset.target_names

print("=" * 60)
print("NAIVE BAYES CLASSIFICATION PROJECT")
print("=" * 60)

print("\nDataset loaded successfully!")
print("Number of rows:", X.shape[0])
print("Number of features:", X.shape[1])
print("Target classes:", list(class_names))


# 3. EXPLORE THE DATA
df = X.copy()
df["target"] = y
df["diagnosis"] = y.map({
    0: "malignant",
    1: "benign"
})

print("\nFirst five rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(X.describe().round(2))

print("\nClass distribution:")
print(df["diagnosis"].value_counts())

print("\nMissing feature values:", X.isnull().sum().sum())

# Save the dataset for further practice
df.to_csv("breast_cancer_dataset.csv", index=False)
print("\nDataset saved as breast_cancer_dataset.csv")


# 4. VISUALIZE THE TARGET DISTRIBUTION
plt.figure(figsize=(7, 4))
sns.countplot(data=df, x="diagnosis")
plt.title("Distribution of Target Classes")
plt.xlabel("Diagnosis label")
plt.ylabel("Number of observations")
plt.tight_layout()
plt.show()


# 5. SPLIT INTO TRAINING AND TESTING DATA
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=RANDOM_STATE,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# 6. CREATE AND TRAIN THE GAUSSIAN NAIVE BAYES MODEL
model = GaussianNB()

print("\nTraining Naive Bayes...")
model.fit(X_train, y_train)
print("Training complete!")


# 7. MAKE PREDICTIONS
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Find the probability column for malignant (class 0)
malignant_column = list(model.classes_).index(0)
malignant_probability = y_proba[:, malignant_column]


# 8. EVALUATE MODEL PERFORMANCE
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label=0)
recall = recall_score(y_test, y_pred, pos_label=0)
f1 = f1_score(y_test, y_pred, pos_label=0)

roc_auc = roc_auc_score(
    (y_test == 0).astype(int),
    malignant_probability
)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision (malignant): {precision:.4f}")
print(f"Recall (malignant):    {recall:.4f}")
print(f"F1-score (malignant):  {f1:.4f}")
print(f"ROC-AUC (malignant):   {roc_auc:.4f}")

print("\nDetailed classification report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=[0, 1],
        target_names=["malignant", "benign"],
        zero_division=0
    )
)


# 9. CONFUSION MATRIX
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

print("\nConfusion matrix:")
print(cm)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["malignant", "benign"]
)
disp.plot(cmap="Blues", values_format="d")
plt.title("Naive Bayes - Confusion Matrix")
plt.tight_layout()
plt.show()


# 10. ROC CURVE
RocCurveDisplay.from_predictions(
    (y_test == 0).astype(int),
    malignant_probability,
    name="Gaussian Naive Bayes"
)
plt.title("ROC Curve: Malignant Class")
plt.tight_layout()
plt.show()

# 11. PREDICT A NEW EXAMPLE
# Use one held-out row to preserve the correct feature structure
new_sample = X_test.iloc[[0]]
actual_label = int(y_test.iloc[0])

predicted_label = int(model.predict(new_sample)[0])
predicted_probabilities = model.predict_proba(new_sample)[0]

print("\n" + "=" * 60)
print("PREDICTION FOR A NEW EXAMPLE")
print("=" * 60)

print("Predicted class:", class_names[predicted_label])
print("Actual class:", class_names[actual_label])

for class_id, probability in zip(
    model.classes_, predicted_probabilities
):
    print(
        f"Probability for {class_names[int(class_id)]}: "
        f"{probability:.4f}"
    )


# 12. OPTIONAL: TUNE THE VARIANCE SMOOTHING PARAMETER
# Run cross-validation on training data only.
print("\nStarting hyperparameter tuning...")

param_grid = {
    "var_smoothing": np.logspace(-12, -7, 6)
}

grid_search = GridSearchCV(
    estimator=GaussianNB(),
    param_grid=param_grid,
    scoring="f1_macro",
    cv=5,
    n_jobs=-1,
    refit=True
)

grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print("\nBest parameters:")
print(grid_search.best_params_)

print("\nBest mean cross-validation macro F1:")
print(f"{grid_search.best_score_:.4f}")

# Evaluate the tuned model on the held-out test set
tuned_predictions = best_model.predict(X_test)

print("\nTuned model test accuracy:")
print(f"{accuracy_score(y_test, tuned_predictions):.4f}")

print("\nTuned model classification report:")
print(
    classification_report(
        y_test,
        tuned_predictions,
        labels=[0, 1],
        target_names=["malignant", "benign"],
        zero_division=0
    )
)


# 13. SAVE THE TRAINED MODEL
joblib.dump(best_model, "naive_bayes_model.joblib")

print("\nModel saved as naive_bayes_model.joblib")
print("\nNaive Bayes project completed successfully!")
