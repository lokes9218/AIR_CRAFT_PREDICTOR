
# Done by Lokeswaran S

# ------------------- PART 1: IMPORTS -------------------
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import xgboost as xgb

# ------------------- PART 2: LOAD DATA -------------------
df = pd.read_csv("AirData.csv")

# ------------------- PART 3: EDA -------------------
print("\n========== BASIC INFO ==========")
print(df.info())

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DESCRIPTIVE STATS ==========")
print(df.describe())

print("\n========== FIRST FEW ROWS ==========")
print(df.head())

# Plot distributions
df.hist(figsize=(12, 8))
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=False, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ------------------- PART 4: HANDLE CATEGORICAL DATA -------------------
label_encoders = {}

for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# ------------------- PART 5: FEATURES + TARGET -------------------
target_column = "Severity"

X = df.drop(columns=[target_column])
y = df[target_column]

# ------------------- PART 6: TRAIN-TEST SPLIT -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------- PART 7: SCALING -------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------- PART 8: VOTING CLASSIFIER -------------------
voting_model = VotingClassifier(
    estimators=[
        ('mlp', MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=500, early_stopping=True)),
        ('rf', RandomForestClassifier(n_estimators=150)),
        ('xgb', xgb.XGBClassifier(eval_metric='mlogloss'))
    ],
    voting='soft'
)

voting_model.fit(X_train_scaled, y_train)
y_pred_vote = voting_model.predict(X_test_scaled)

print("\n========== SOFT VOTING RESULTS ==========")
print("Accuracy:", accuracy_score(y_test, y_pred_vote))
print("\nClassification Report:\n", classification_report(y_test, y_pred_vote))

# ------------------- PART 9: HYPERPARAMETER TUNING (XGBOOST) -------------------
param_grid = {
    "n_estimators": [50, 100, 150, 200],
    "max_depth": [3, 4, 5, 6],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.8, 1.0]
}

xgb_model = xgb.XGBClassifier(eval_metric='mlogloss')

random_search = RandomizedSearchCV(
    xgb_model,
    param_distributions=param_grid,
    n_iter=10,
    cv=3,
    scoring='accuracy',
    verbose=1,
    random_state=42
)

random_search.fit(X_train_scaled, y_train)

print("\n========== BEST PARAMETERS ==========")
print(random_search.best_params_)
print("\n========== BEST SCORE ==========")
print(random_search.best_score_)
