# Neural Network Models

This repository contains implementations of various neural network models for solving classification and prediction problems. These models leverage deep learning techniques such as CNNs, MLPs, RBF networks, and LSTMs to demonstrate their functionality and use cases.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Usage](#usage)
6. [Project Structure](#project-structure)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

---

## Project Overview
The goal of this project is to explore and implement various neural network architectures and apply them to different problem domains, including:
- Image classification (binary classification using CNNs).
- Predicting customer churn using MLPs.
- Implementing custom Radial Basis Function (RBF) networks.
- Forecasting time series data with LSTMs.

Each model is implemented in Python and leverages popular libraries such as TensorFlow and Keras.

---

## Features
- **Image Classification**: Classifies images as either belonging to one of two categories.
- **Customer Churn Prediction**: Predicts whether customers are likely to churn using structured data.
- **RBF Networks**: Implements custom RBF layers for nonlinear classification.
- **Time Series Forecasting**: Uses LSTM layers to predict future values based on historical data.

---

## Requirements
The project requires the following Python libraries:
- `tensorflow`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`

To install all dependencies, use the `requirements.txt` file provided.

---

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Harshita-2205/Neural-Network-Models.git
   cd Neural-Network-Models
   ```
2. (Optional) Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage
1. Prepare your datasets:
   - For image classification, organize your images in a folder structure for training and testing.
   - For churn prediction, ensure the dataset has a `Churn` column.
   - For time series forecasting, provide a CSV with relevant columns like `close_price`.

2. Run the desired script:
   - **Image Classification**:
     ```bash
     python image_classification.py
     ```
   - **Customer Churn Prediction**:
     ```bash
     python churn_prediction.py
     ```
   - **RBF Neural Network**:
     ```bash
     python rbf_network.py
     ```
   - **Time Series Prediction**:
     ```bash
     python time_series_prediction.py
     ```

3. Results will be printed in the console or saved as plots in the `outputs/` folder (if implemented).

---

## Project Structure
```
Neural-Network-Models/
│
├── datasets/                  # Datasets used in the project
├── models/                    # Saved models (if applicable)
├── outputs/                   # Generated results like plots or logs
├── image_classification.py    # CNN-based image classification script
├── churn_prediction.py        # Customer churn prediction script
├── rbf_network.py             # RBF network implementation
├── time_series_prediction.py  # LSTM for time series forecasting
├── requirements.txt           # Project dependencies
├── LICENSE                    # License file
└── README.md                  # Project documentation
```

---

## Results
### Image Classification
- Accuracy and loss plots for training and validation.
- Final test accuracy on unseen data.

### Customer Churn Prediction
- Metrics such as accuracy, precision, recall, and F1-score.
- Confusion matrix visualization.

### RBF Neural Network
- Decision boundary visualization (if applicable).
- Performance metrics.

### Time Series Prediction
- Line plots comparing actual vs. predicted values.
- Mean Squared Error (MSE) for evaluation.

---

## Contributing
Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature/your-feature
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add your message here"
   ```
4. Push to your fork:
   ```bash
   git push origin feature/your-feature
   ```
5. Open a pull request.

---

## License
This project is licensed under the MIT License. See `LICENSE` for more details.
