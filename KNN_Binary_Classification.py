import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
data = pd.read_csv("Audio_Features_Classification.csv", sep=';')

# Extract features and labels
X = data[['Spectral Centroid', 'Zero-Crossing Rate', 'Spectral Flux']].values
y = data['Label'].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Define the k-NN model
knn = KNeighborsClassifier()

# Define the parameter grid for RandomizedSearchCV
"""param_distributions = {
    'n_neighbors': range(1, 50),  # Test different numbers of neighbors
    'weights': ['uniform', 'distance'],  # Compare uniform and distance weighting
    'p': [1, 2]  # Include Manhattan (p=1) and Euclidean (p=2) distances
}"""


param_distributions = {
    'n_neighbors': [4],  # Test different numbers of neighbors
    'weights': ['distance'],  # Compare uniform and distance weighting
    'p': [2]  # Include Manhattan (p=1) and Euclidean (p=2) distances
}

# Perform Randomized Search
random_search = RandomizedSearchCV(
    knn,
    param_distributions=param_distributions,
    n_iter=150,
    scoring='accuracy',
    cv=10,
    random_state=42,
    verbose=1
)

# Fit the model
random_search.fit(X_train, y_train)

# Print the best parameters
print("Best Parameters:")
print(random_search.best_params_)

# Train the best model
best_knn = random_search.best_estimator_
best_knn.fit(X_train, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(best_knn, X_train, y_train, cv=10, scoring='accuracy')
print("Cross-validation Accuracy:", np.mean(cv_scores))

# Make predictions
y_pred = best_knn.predict(X_test)

# Compute and print metrics
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Percussion', 'Harmonic'],
            yticklabels=['Percussion', 'Harmonic'])
plt.title("Classification Results")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

train_errors = 1 - cross_val_score(best_knn, X_train, y_train, cv=10, scoring='accuracy')  
test_errors = 1 - cross_val_score(best_knn, X_test, y_test, cv=10, scoring='accuracy')

plt.figure(figsize=(12, 6))
plt.plot(range(1, 11), train_errors, label="Training Error", marker='o')
plt.plot(range(1, 11), test_errors, label="Test Error", marker='o')
plt.xticks(range(1, 11))
plt.xlabel("Fold")
plt.ylabel("Error")
plt.title("Training and Test Error (10 Folds)")
plt.legend()
plt.grid(True)
plt.show()


# 3D Plot of the k-NN decision boundaries
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot training points
ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c=y_train, cmap='viridis', label='Training Data', alpha=0.6)

# Plot test points
ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=y_test, cmap='coolwarm', marker='x', label='Test Data', alpha=0.8)

ax.set_title("3D k-NN Classification")
ax.set_xlabel("Spectral Centroid")
ax.set_ylabel("Zero-Crossing Rate")
ax.set_zlabel("Spectral Flux")

plt.legend()
plt.show()