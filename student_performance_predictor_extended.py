import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# STEP 1: Load Extended Dataset

data = pd.read_csv("student_data_extended.csv")

# STEP 2: Encode Target Column

le = LabelEncoder()
data["Final Grade Encoded"] = le.fit_transform(data["Final Grade"])


# STEP 3: Prepare Features and Target

X = data[["Attendance (%)", "Internal Marks", "Assignment Score", "Peer Support Score"]]
y = data["Final Grade Encoded"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# STEP 4: Model Training

model = DecisionTreeClassifier(random_state=42, max_depth=5)
model.fit(X_train, y_train)

# STEP 5: Predictions and Evaluation

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))


# STEP 6: Visual Analysis

plt.figure(figsize=(7,5))
plt.scatter(
    data["Peer Support Score"],
    data["Internal Marks"],
    c=data["Final Grade Encoded"],
    cmap="plasma",
    s=70,
    edgecolor="k",
    alpha=0.8
)
plt.xlabel("Peer Support Score (1–10)")
plt.ylabel("Internal Marks")
plt.title("Effect of Peer Support on Student Performance")
plt.colorbar(label="Performance Level (Encoded)")
plt.show()


# STEP 7: Correlation Matrix

print("\nCorrelation Matrix:")
print(data[["Attendance (%)", "Internal Marks", "Assignment Score", "Peer Support Score"]].corr())
