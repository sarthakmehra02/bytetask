import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer # type: ignore
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Load the dataset
df = pd.read_csv('Task_1.csv')

# Check for missing values and remove them if any
df.dropna(inplace=True)

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['labels'], test_size=0.2, random_state=42)

# Convert text data into numerical features using TfidfVectorizer
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the Logistic Regression model
model = LogisticRegression()

# Train the model
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],  # Inverse of regularization strength
    'penalty': ['l2'],  # Regularization
    'solver': ['lbfgs']  # Optimizer
}

grid_search = GridSearchCV(LogisticRegression(), param_grid, cv=5)
grid_search.fit(X_train_tfidf, y_train)

# Best parameters and the corresponding accuracy
print("Best Parameters:", grid_search.best_params_)

# Train the model with best parameters
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_tfidf)

# Evaluate the tuned model
best_accuracy = accuracy_score(y_test, y_pred_best)
best_conf_matrix = confusion_matrix(y_test, y_pred_best)

print(f"Tuned Model Accuracy: {best_accuracy}")
print(f"Tuned Model Confusion Matrix:\n{best_conf_matrix}")
