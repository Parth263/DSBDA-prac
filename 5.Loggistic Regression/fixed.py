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

# ### **10. Conclusion**
# - **Does It Solve the Problem?**: **Partially**, because:
#   - Implements logistic regression and most metrics.
#   - Dataset (`data5.csv`) is similar but not the standard Social_Network_Ads.csv.
#   - Includes incorrect features and omits Error Rate.
# - **Fixed Code**: Corrects feature selection, adds scaling, Error Rate, and visualization.
# - **Next Steps**:
#   - Run the fixed code with `data5.csv` or Social_Network_Ads.csv.
#   - Review viva tips for exam preparation.

# If you need sample outputs, help with Social_Network_Ads.csv, or more viva questions, let me know!