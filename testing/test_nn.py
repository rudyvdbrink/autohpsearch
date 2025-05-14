import numpy as np
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from autohpsearch.models.nn import AutoHPSearchClassifier, AutoHPSearchRegressor

# Set random seed for reproducibility
np.random.seed(42)

# ---- Classification Example ----
print("Classification Example:")
# Generate synthetic classification data
X_clf, y_clf = make_classification(
    n_samples=1000, 
    n_features=10, 
    n_informative=5, 
    n_redundant=2, 
    n_classes=2, 
    random_state=42
)

# Split data
X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
    X_clf, y_clf, test_size=0.2, random_state=42
)

# Scale features
scaler_clf = StandardScaler()
X_train_clf = scaler_clf.fit_transform(X_train_clf)
X_test_clf = scaler_clf.transform(X_test_clf)

# Create and train the classifier
clf = AutoHPSearchClassifier(
    hidden_layers=(32, 16),
    activation='relu',
    dropout_rate=0.2,
    learning_rate=0.001,
    optimizer='adam',
    batch_size=32,
    epochs=50
)

# Enable verbose mode
clf.verbose = True
clf.fit(X_train_clf, y_train_clf)

# Make predictions
y_pred_clf = clf.predict(X_test_clf)
y_proba_clf = clf.predict_proba(X_test_clf)

# Evaluate performance
clf_accuracy = accuracy_score(y_test_clf, y_pred_clf)
print(f"Classification accuracy: {clf_accuracy:.4f}")

# ---- Regression Example ----
print("\nRegression Example:")
# Generate synthetic regression data
X_reg, y_reg = make_regression(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    noise=0.1,
    random_state=42
)

# Split data
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42
)

# Scale features
scaler_reg = StandardScaler()
X_train_reg = scaler_reg.fit_transform(X_train_reg)
X_test_reg = scaler_reg.transform(X_test_reg)

# Create and train the regressor
reg = AutoHPSearchRegressor(
    hidden_layers=(32, 16),
    activation='relu',
    dropout_rate=0.1,  # Typically lower dropout for regression
    learning_rate=0.001,
    optimizer='adam',
    batch_size=32,
    epochs=50
)

# Enable verbose mode
reg.verbose = True
reg.fit(X_train_reg, y_train_reg)

# Make predictions
y_pred_reg = reg.predict(X_test_reg)

# Evaluate performance
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)
print(f"Regression MSE: {mse:.4f}")
print(f"Regression R²: {r2:.4f}")

# Optional: Plot regression predictions vs true values
plt.figure(figsize=(10, 6))
plt.scatter(y_test_reg, y_pred_reg, alpha=0.5)
plt.plot([-3, 3], [-3, 3], 'r--')  # Perfect prediction line
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Regression Predictions vs True Values')
plt.grid(True)
plt.show()