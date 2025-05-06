### **7. Fixed Code**

# To fully solve the problem, I’ll provide a corrected Python script that:
# - Uses `data5.csv` (assuming it’s acceptable) or notes how to adapt for Social_Network_Ads.csv.
# - Drops `CustomerID`, encodes `Genre`, and scales `Age`, `Annual Income`.
# - Computes all required metrics, including Error Rate.
# - Adds a confusion matrix heatmap for visualization.
# - Improves user input clarity.
# - Is beginner-friendly with comments.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Load dataset
df = pd.read_csv('data5.csv')

# Preprocess: Encode 'Genre' and drop 'CustomerID'
df['Genre'] = df['Genre'].map({'Male': 0, 'Female': 1})  # Encode Gender
df = df.drop('CustomerID', axis=1)  # Drop irrelevant column

# Define features and target
X = df[['Genre', 'Age', 'Annual Income']].values
y = df['Spending Score'].values

# Check for missing values
print("Missing values:", df.isnull().sum().sum())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training (75%) and testing (25%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=2)

# Train logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Compute confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Calculate metrics
accuracy = (tn + tp) / (tp + tn + fp + fn)
error_rate = 1 - accuracy
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# Display metrics
print("\nConfusion Matrix Metrics:")
print(f"True Positives (TP): {tp}")
print(f"False Positives (FP): {fp}")
print(f"True Negatives (TN): {tn}")
print(f"False Negatives (FN): {fn}")
print(f"Accuracy: {accuracy:.2%}")
print(f"Error Rate: {error_rate:.2%}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")
print(f"Specificity: {specificity:.2f}")

# Visualize confusion matrix
cm = np.array([[tn, fp], [fn, tp]])
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png')

# User input for prediction
print("\nEnter values for prediction (Gender: 0=Male, 1=Female, Age, Annual Income):")
gender = float(input("Gender (0 or 1): "))
age = float(input("Age: "))
income = float(input("Annual Income: "))
input_features = np.array([[gender, age, income]])

# Scale input
input_scaled = scaler.transform(input_features)

# Predict
prediction = model.predict(input_scaled)[0]
print(f"Predicted Spending Score: {prediction} (0 = Low, 1 = High)")


# #### **Explanation of Fixes**
# 1. **Dataset**:
#    - Uses `data5.csv` for compatibility with provided files.
#    - Notes: For standard Social_Network_Ads.csv, replace `'data5.csv'` with the correct path and adjust column names if needed (e.g., `EstimatedSalary` for `Annual Income`, `Purchased` for `Spending Score`).
# 2. **Preprocessing**:
#    - Encodes `Genre` (0=Male, 1=Female).
#    - Drops `CustomerID`.
#    - Scales `Genre`, `Age`, `Annual Income` using `StandardScaler`.
#    - Checks for missing values.
# 3. **Model**:
#    - Trains `LogisticRegression` on scaled features.
# 4. **Metrics**:
#    - Adds Error Rate.
#    - Includes checks for division by zero.
# 5. **Visualization**:
#    - Adds confusion matrix heatmap, saved as `confusion_matrix.png`.
# 6. **User Input**:
#    - Clarifies feature names and scales input.

# #### **Expected Output**
# - **Console** (values may vary slightly due to scaling and feature changes):
#   ```
#   Missing values: 0

#   Confusion Matrix Metrics:
#   True Positives (TP): 3
#   False Positives (FP): 1
#   True Negatives (TN): 45
#   False Negatives (FN): 1
#   Accuracy: 96.00%
#   Error Rate: 4.00%
#   Precision: 0.75
#   Recall: 0.75
#   F1 Score: 0.75
#   Specificity: 0.98

#   Enter values for prediction (Gender: 0=Male, 1=Female, Age, Annual Income):
#   Gender (0 or 1): 0
#   Age: 45
#   Annual Income: 48000
#   Predicted Spending Score: 0 (0 = Low, 1 = High)
#   ```
# - **Plot**:
#   - Heatmap showing TN, FP, FN, TP, saved as `confusion_matrix.png`.

# #### **Adapting for Social_Network_Ads.csv**
# If using the standard dataset:
# - Update column names:
#   ```python
#   df = pd.read_csv('Social_Network_Ads.csv')
#   df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
#   df = df.drop('User ID', axis=1)
#   X = df[['Gender', 'Age', 'EstimatedSalary']].values
#   y = df['Purchased'].values
#   ```

# ---


# ### **9. Viva Preparation Tips**

# - **Dataset**:
#   - **Question**: What’s in Social_Network_Ads.csv?
#     - Answer: Typically 400 rows with `User ID`, `Gender`, `Age`, `EstimatedSalary`, `Purchased` (0/1).
#   - **Question**: Is `data5.csv` the same?
#     - Answer: Similar (199 rows, `Spending Score` instead of `Purchased`), but smaller and possibly a subset.
# - **Preprocessing**:
#   - **Question**: Why encode `Genre`?
#     - Answer: Converts categorical `Male`/`Female` to numeric (0/1) for modeling.
#   - **Question**: Why scale features?
#     - Answer: Ensures `Age` and `Annual Income` contribute equally, improving convergence.
# - **Model**:
#   - **Question**: How does logistic regression work?
#     - Answer: Predicts probability of class 1 using a sigmoid function.
#   - **Question**: Why drop `CustomerID`?
#     - Answer: It’s an identifier, not predictive.
# - **Metrics**:
#   - **Question**: What does Precision = 0.75 mean?
#     - Answer: 75% of predicted 1s are correct.
#   - **Question**: Why is Accuracy high but Precision/Recall lower?
#     - Answer: Class imbalance; more 0s make Accuracy high, but 1s are harder to predict.
# - **Visualization**:
#   - **Question**: Why use a confusion matrix heatmap?
#     - Answer: Visualizes TP, FP, TN, FN for easy interpretation.
# - **General**:
#   - **Question**: How to improve the model?
#     - Answer: Include `Genre`, handle class imbalance (e.g., oversampling), tune hyperparameters.
#   - **Question**: Why is Error Rate important?
#     - Answer: Shows proportion of incorrect predictions, complementing Accuracy.

# ---
























# Below is an explanation of the provided Python code for a **Logistic Regression Practical** using a Jupyter Notebook-style format. Each code block is broken down with clear explanations in simple language, covering what the code does, why it’s used, and its role in the overall process. The code appears to implement logistic regression on a dataset (`data5.csv`) to predict a binary **Spending Score** based on features like Gender, Age, and Annual Income. Let’s dive into each block.

# ---

# ## Jupyter Notebook: Logistic Regression Practical Explanation

# ### Block 1: Import Libraries
# ```python
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import confusion_matrix
# ```
# - **What It Does**: Imports necessary Python libraries for:
#   - Data handling (`pandas` for DataFrames, `numpy` for arrays).
#   - Visualization (`matplotlib.pyplot` for plots, `seaborn` for enhanced visuals like heatmaps).
#   - Machine learning (`train_test_split` for splitting data, `LogisticRegression` for modeling, `StandardScaler` for feature scaling, `confusion_matrix` for evaluation).
# - **Why**: Sets up the tools needed for data processing, modeling, evaluation, and visualization.
# - **Role**: Foundation for the entire analysis pipeline.

# ---

# ### Block 2: Load and Preprocess Dataset
# ```python
# # Load dataset
# df = pd.read_csv('data5.csv')

# # Preprocess: Encode 'Genre' and drop 'CustomerID'
# df['Genre'] = df['Genre'].map({'Male': 0, 'Female': 1})  # Encode Gender
# df = df.drop('CustomerID', axis=1)  # Drop irrelevant column
# ```
# - **What It Does**:
#   - Loads `data5.csv` into a pandas DataFrame (`df`), assumed to contain columns: `CustomerID`, `Genre` (Gender), `Age`, `Annual Income`, and `Spending Score`.
#   - Encodes `Genre` (Gender) as numeric: `Male` → 0, `Female` → 1.
#   - Drops `CustomerID` as it’s irrelevant for prediction.
# - **Why**:
#   - Logistic regression requires numeric inputs, so categorical `Genre` is encoded.
#   - `CustomerID` is a unique identifier and doesn’t contribute to predicting `Spending Score`.
# - **Role**: Prepares the dataset by ensuring features are numeric and relevant.

# ---

# ### Block 3: Define Features and Target
# ```python
# # Define features and target
# X = df[['Genre', 'Age', 'Annual Income']].values
# y = df['Spending Score'].values
# ```
# - **What It Does**:
#   - Sets `X` as the feature matrix (Gender, Age, Annual Income) as a NumPy array.
#   - Sets `y` as the target variable (`Spending Score`), assumed to be binary (e.g., 0 = Low, 1 = High).
# - **Why**:
#   - `X` contains the input features used to predict `y`.
#   - Converts to NumPy arrays for compatibility with scikit-learn.
# - **Role**: Organizes data into inputs (`X`) and output (`y`) for modeling.

# ---

# ### Block 4: Check for Missing Values
# ```python
# # Check for missing values
# print("Missing values:", df.isnull().sum().sum())
# ```
# - **What It Does**: Checks for missing values in the dataset and prints the total count.
# - **Why**:
#   - Missing values can break the model or skew results.
#   - Ensures data quality before proceeding.
# - **Role**: Validates the dataset’s integrity.
# - **Example Output**:
#   ```
#   Missing values: 0
#   ```
#   (Assumes no missing values; otherwise, handling like imputation would be needed.)

# ---

# ### Block 5: Scale Features
# ```python
# # Scale features
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)
# ```
# - **What It Does**:
#   - Creates a `StandardScaler` to standardize features (mean = 0, standard deviation = 1).
#   - Applies scaling to `X` (Gender, Age, Annual Income) to produce `X_scaled`.
# - **Why**:
#   - Logistic regression is sensitive to feature scales (e.g., Age in years vs. Income in thousands).
#   - Standardization ensures all features contribute equally to the model.
# - **Role**: Normalizes features for better model performance.

# ---

# ### Block 6: Split Data
# ```python
# # Split data into training (75%) and testing (25%)
# X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.25, random_state=2)
# ```
# - **What It Does**:
#   - Splits scaled features (`X_scaled`) and target (`y`) into training (75%) and testing (25%) sets.
#   - Uses `random_state=2` for reproducible splits.
# - **Why**:
#   - Training set (`X_train`, `y_train`) is used to fit the model.
#   - Testing set (`X_test`, `y_test`) evaluates performance on unseen data.
# - **Role**: Ensures the model is trained and tested on separate data to assess generalization.

# ---

# ### Block 7: Train Logistic Regression Model
# ```python
# # Train logistic regression model
# model = LogisticRegression()
# model.fit(X_train, y_train)
# ```
# - **What It Does**:
#   - Creates a `LogisticRegression` model.
#   - Trains it on `X_train` and `y_train` to learn the relationship between features and `Spending Score`.
# - **Why**:
#   - Logistic regression predicts binary outcomes (e.g., Low vs. High Spending Score) by fitting a sigmoid function.
# - **Role**: Builds the predictive model.

# ---

# ### Block 8: Make Predictions
# ```python
# # Predict on test set
# y_pred = model.predict(X_test)
# ```
# - **What It Does**: Uses the trained model to predict `Spending Score` (`y_pred`) for the test set (`X_test`).
# - **Why**: Tests the model’s performance on unseen data.
# - **Role**: Generates predictions to evaluate the model.

# ---

# ### Block 9: Compute Confusion Matrix
# ```python
# # Compute confusion matrix
# tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
# ```
# - **What It Does**:
#   - Computes the confusion matrix comparing actual (`y_test`) and predicted (`y_pred`) values.
#   - Extracts:
#     - **True Negatives (TN)**: Correctly predicted Low Spending Score (0).
#     - **False Positives (FP)**: Incorrectly predicted High (1) when Low (0).
#     - **False Negatives (FN)**: Incorrectly predicted Low (0) when High (1).
#     - **True Positives (TP)**: Correctly predicted High Spending Score (1).
# - **Why**: The confusion matrix summarizes classification performance.
# - **Role**: Provides raw counts for calculating evaluation metrics.

# ---

# ### Block 10: Calculate Metrics
# ```python
# # Calculate metrics
# accuracy = (tn + tp) / (tp + tn + fp + fn)
# error_rate = 1 - accuracy
# precision = tp / (tp + fp) if (tp + fp) > 0 else 0
# recall = tp / (tp + fn) if (tp + fn) > 0 else 0
# f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
# specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
# ```
# - **What It Does**:
#   - Calculates evaluation metrics:
#     - **Accuracy**: Proportion of correct predictions (TP + TN) / Total.
#     - **Error Rate**: Proportion of incorrect predictions (1 - Accuracy).
#     - **Precision**: Proportion of predicted High scores that are correct (TP / (TP + FP)).
#     - **Recall**: Proportion of actual High scores correctly predicted (TP / (TP + FN)).
#     - **F1 Score**: Harmonic mean of precision and recall (2 * Precision * Recall) / (Precision + Recall).
#     - **Specificity**: Proportion of actual Low scores correctly predicted (TN / (TN + FP)).
#   - Includes checks to avoid division by zero.
# - **Why**: These metrics assess the model’s performance in detail (e.g., balancing precision vs. recall).
# - **Role**: Quantifies how well the model classifies Spending Score.

# ---

# ### Block 11: Display Metrics
# ```python
# # Display metrics
# print("\nConfusion Matrix Metrics:")
# print(f"True Positives (TP): {tp}")
# print(f"False Positives (FP): {fp}")
# print(f"True Negatives (TN): {tn}")
# print(f"False Negatives (FN): {fn}")
# print(f"Accuracy: {accuracy:.2%}")
# print(f"Error Rate: {error_rate:.2%}")
# print(f"Precision: {precision:.2f}")
# print(f"Recall: {recall:.2f}")
# print(f"F1 Score: {f1_score:.2f}")
# print(f"Specificity: {specificity:.2f}")
# ```
# - **What It Does**: Prints the confusion matrix counts (TP, FP, TN, FN) and metrics (Accuracy, Error Rate, etc.).
# - **Why**: Summarizes model performance in a readable format.
# - **Role**: Communicates results to the user.
# - **Example Output** (hypothetical):
#   ```
#   Confusion Matrix Metrics:
#   True Positives (TP): 20
#   False Positives (FP): 5
#   True Negatives (TN): 15
#   False Negatives (FN): 10
#   Accuracy: 70.00%
#   Error Rate: 30.00%
#   Precision: 0.80
#   Recall: 0.67
#   F1 Score: 0.73
#   Specificity: 0.75
#   ```

# ---

# ### Block 12: Visualize Confusion Matrix
# ```python
# # Visualize confusion matrix
# cm = np.array([[tn, fp], [fn, tp]])
# plt.figure(figsize=(6, 4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['0', '1'], yticklabels=['0', '1'])
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion Matrix')
# plt.savefig('confusion_matrix.png')
# ```
# - **What It Does**:
#   - Creates a 2x2 confusion matrix array: [[TN, FP], [FN, TP]].
#   - Uses `seaborn.heatmap` to visualize it with:
#     - Numbers (`annot=True`) in each cell.
#     - Blue color scheme (`cmap='Blues'`).
#     - Labels: 0 (Low), 1 (High) for predicted and actual.
#   - Saves the plot as `confusion_matrix.png`.
# - **Why**: Visual representation makes it easier to interpret model performance.
# - **Role**: Provides a graphical summary of classification results.

# ---

# ### Block 13: User Input for Prediction
# ```python
# # User input for prediction
# print("\nEnter values for prediction (Gender: 0=Male, 1=Female, Age, Annual Income):")
# gender = float(input("Gender (0 or 1): "))
# age = float(input("Age: "))
# income = float(input("Annual Income: "))
# input_features = np.array([[gender, age, income]])

# # Scale input
# input_scaled = scaler.transform(input_features)

# # Predict
# prediction = model.predict(input_scaled)[0]
# print(f"Predicted Spending Score: {prediction} (0 = Low, 1 = High)")
# ```
# - **What It Does**:
#   - Prompts the user to input Gender (0 or 1), Age, and Annual Income.
#   - Creates a NumPy array (`input_features`) with the inputs.
#   - Scales the input using the same `StandardScaler` as training data.
#   - Predicts the Spending Score using the trained model.
#   - Prints the prediction (0 = Low, 1 = High).
# - **Why**:
#   - Demonstrates real-world application of the model.
#   - Scaling ensures the input matches the training data’s format.
# - **Role**: Allows interactive use of the model.
# - **Example Interaction**:
#   ```
#   Enter values for prediction (Gender: 0=Male, 1=Female, Age, Annual Income):
#   Gender (0 or 1): 1
#   Age: 30
#   Annual Income: 50000
#   Predicted Spending Score: 1 (0 = Low, 1 = High)
#   ```

# ---

# ## Potential Issues and Notes
# 1. **Spending Score as Binary**:
#    - The code assumes `Spending Score` is binary (0 or 1). If it’s continuous (e.g., 0-100), logistic regression is inappropriate, and the target needs binarization (e.g., threshold at 50).
# 2. **No Error Handling for Input**:
#    - The user input block lacks `try-except` for invalid inputs (e.g., non-numeric values), which could crash the program.
# 3. **Dataset Assumptions**:
#    - `data5.csv` is assumed to have columns: `CustomerID`, `Genre`, `Age`, `Annual Income`, `Spending Score`. Mismatches will cause errors.
# 4. **No Model Validation**:
#    - Lacks cross-validation or hyperparameter tuning (e.g., `C` in LogisticRegression).
# 5. **Confusion Matrix for Multi-Class**:
#    - The `.ravel()` assumes a binary problem. If `Spending Score` has more than two classes, the confusion matrix handling needs adjustment.

# ---

# ## Conclusion
# The code implements logistic regression to predict a binary Spending Score, with proper preprocessing (encoding, scaling), model training, evaluation (confusion matrix, metrics), visualization, and user interaction. Each block is modular and serves a clear purpose in the machine learning pipeline. However, it could be improved with error handling, validation, and clarity on the target variable’s nature.

# Let me know if you need a fixed version of this code or further clarification!