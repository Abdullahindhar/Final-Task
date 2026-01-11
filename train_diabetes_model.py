# =========================
# 1. Imports
# =========================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, confusion_matrix

# =========================
# 2. Load Dataset
# =========================
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

columns = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
]

df = pd.read_csv(url, names=columns)

# =========================
# 3. Features & Target
# =========================
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

print("\nFirst 5 Rows:\n", df.head())
print("\nSummary Statistics:\n", df.describe())
print("\nClass Distribution:\n", y.value_counts())

# =========================
# 4. Handle Invalid Zero Values
# =========================
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# =========================
# 5. Scaling
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# 6. Stratified Split (80/20)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# =========================
# 7. PCA Analysis
# =========================
pca = PCA()
pca.fit(X_train)

cumulative_variance = pca.explained_variance_ratio_.cumsum()

components_95 = np.argmax(cumulative_variance >= 0.95) + 1
components_99 = np.argmax(cumulative_variance >= 0.99) + 1

print("\nPCA Components for 95% variance:", components_95)
print("PCA Components for 99% variance:", components_99)

# Apply PCA (95%)
pca_95 = PCA(n_components=components_95)
X_train_pca = pca_95.fit_transform(X_train)
X_test_pca = pca_95.transform(X_test)

print("\nExplained Variance Ratio:\n", pca_95.explained_variance_ratio_)

# =========================
# 8. Model Training & Evaluation
# =========================
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(kernel="rbf", probability=True)
}

best_model = None
best_accuracy = 0

for name, model in models.items():
    model.fit(X_train_pca, y_train)
    y_pred = model.predict(X_test_pca)

    acc = accuracy_score(y_test, y_pred)

    print(f"\n{name}")
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model

print("\nBest Model:", best_model)
print("Best Accuracy:", best_accuracy)

# =========================
# 9. Save Model & Objects
# =========================
joblib.dump(best_model, "diabetes_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(pca_95, "pca.pkl")

print("\n✅ Model, Scaler, and PCA saved successfully!")
