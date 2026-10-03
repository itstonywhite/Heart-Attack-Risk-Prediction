import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Set visual style
sns.set_theme(style="whitegrid")

# Load dataset
df = pd.read_csv('heart.csv')

# View dataset dimensions and preview
# print(f"Dataset Shape: {df.shape}")
# print("\nFirst 5 rows:")
# print(df.head())

# Summary statistics and info
# print("\nDataset Info:")
# df.info()

# print("\nMissing Values:")
# print(df.isnull().sum())

# Remove duplicate rows
duplicate_count = df.duplicated().sum()
# print(f"\nNumber of duplicate rows found: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates().reset_index(drop=True)
    # print("Duplicates removed successfully.")
    
# Separate features and target
X = df.drop('output', axis=1)
y = df['output']

# Split data: 80% Training, 20% Testing (stratified by target class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize algorithms
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42)
}

# Dictionary to store performance metrics
results = {}

# Train and evaluate each model
for name, model in models.items():
    # Fit model
    model.fit(X_train_scaled, y_train)
    
    # Predict on test data
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Store results
    results[name] = {
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1-Score': f1,
        'Confusion Matrix': cm,
        'Model Object': model
    }

# Convert evaluation metrics into a readable DataFrame
results_df = pd.DataFrame(results).T.drop(columns=['Confusion Matrix', 'Model Object'])
# print("--- Model Performance Comparison ---")
# print(results_df.round(4))

# Plot evaluation metrics bar chart
ax = results_df.plot(kind='bar', figsize=(10, 6), width=0.8)
plt.title('Performance Comparison of Classification Models', fontsize=14)
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
# plt.show() # Renders the plot

# print("-"*100)

# Plot Confusion Matrices for all models
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for idx, (name, metrics) in enumerate(results.items()):
    sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
    axes[idx].set_title(f'{name} Confusion Matrix')
    axes[idx].set_xlabel('Predicted Label')
    axes[idx].set_ylabel('True Label')

plt.tight_layout()
# plt.show() # Renders the plot

# Identify best model using F1-Score
best_model_name = results_df['F1-Score'].astype(float).idxmax()
best_model = results[best_model_name]['Model Object']

# print(f"Top Performing Model: {best_model_name}")
# print(f"F1-Score: {results_df.loc[best_model_name, 'F1-Score']:.4f}")

# Save the best model and scaler to disk
joblib.dump(best_model, 'best_heart_disease_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# print("\nModel artifact ('best_heart_disease_model.pkl') and scaler ('scaler.pkl') saved successfully!")


# Tony White ✍️