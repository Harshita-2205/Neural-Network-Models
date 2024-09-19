import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K

# Load the customer churn dataset
data = pd.read_csv('MOCK_DATA (3).csv')  # Replace with your dataset

# Assuming 'Churn' column contains 'Yes' or 'No', convert it to binary (1 for 'Yes', 0 for 'No')
data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})

# Convert categorical columns to numerical 
data = pd.get_dummies(data, drop_first=True)

# Separate features and target
X = data.drop(columns=['Churn'])
y = data['Churn']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Custom RBF Layer Definition
class RBFLayer(layers.Layer):
    def __init__(self, units, gamma):
        super(RBFLayer, self).__init__()
        self.units = units
        self.gamma = K.cast_to_floatx(gamma)

    def build(self, input_shape):
        self.mu = self.add_weight(name='mu', shape=(self.units, input_shape[1]), initializer='uniform', trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = K.expand_dims(inputs, axis=1) - self.mu
        return K.exp(-self.gamma * K.sum(K.square(diff), axis=-1))

# RBF Neural Network
model = models.Sequential()

# Input layer
model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))

# Add custom RBF layer with 10 units and a gamma value of 0.5
model.add(RBFLayer(units=10, gamma=0.5))

# Dense layer for classification
model.add(layers.Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy}")

# Predictions
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

# Print classification report
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Plot the training and validation accuracy over epochs
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
