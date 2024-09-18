import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Load the customer churn dataset
data = pd.read_csv('MOCK_DATA (3).csv')  # Replace with your dataset
print(data.head())

# Assuming 'Churn' column contains 'Yes' or 'No', convert it to binary (1 for 'Yes', 0 for 'No')
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
print(data['Churn'].value_counts())  # Check if the conversion worked correctly

# Convert categorical columns to numerical (e.g., One-Hot Encoding)
data = pd.get_dummies(data, drop_first=True)  # Automatically encodes categorical features
print(data.head())

# Separate features and target
X = data.drop(columns=['Churn'])  # Drop the 'Churn' column from the features
y = data['Churn']  # 'Churn' is the target column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the MLP model
model = Sequential()

# Add fully connected (Dense) layers
model.add(Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)))  # Input layer
model.add(Dense(units=64, activation='relu'))  # Hidden layer
model.add(Dense(units=1, activation='sigmoid'))  # Output layer for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Make predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)  # Convert probabilities to binary (0 or 1)

# Print classification report
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
