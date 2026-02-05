# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the program.
2. Data preprocessing: Cleanse data,handle missing values,encode categorical variables.
3. Model Training: Fit logistic regression model on preprocessed data.
4. Model Evaluation: Assess model performance using metrics like accuracyprecisioon,recall.
5. Prediction: Predict placement status for new student data using trained model.
6. End of program.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: udhayamoorthy A
RegisterNumber:  212225040477
*/
```
```
import pandas as pd
import numpy as np

data = pd.read_csv("/content/Placement_Data.csv")
data1 = data.drop(['sl_no', 'salary'], axis=1)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])

X = data1.iloc[:, :-1].values  # Features
Y = data1["status"].values  # Target variable

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

theta = np.random.randn(X.shape[1])  # Random initialization
alpha = 0.01  # Learning rate
num_iterations = 1000  # Number of iterations

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h + 1e-15) + (1 - y) * np.log(1 - h + 1e-15)) / len(y)

def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta, X, Y, alpha, num_iterations)

def predict(theta, X):
    h = sigmoid(X.dot(theta))
    return np.where(h >= 0.5, 1, 0)

y_pred = predict(theta, X)
accuracy = np.mean(y_pred == Y)

print("Accuracy:", accuracy)
print("\nPredicted:\n", y_pred)
print("\nActual:\n", Y)

xnew = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])  # Example input
xnew = scaler.transform(xnew)  # Apply same scaling as training data
y_prednew = predict(theta, xnew)
print("\nPredicted Result:", y_prednew)
```

## Output:

<img width="770" height="416" alt="543196340-21afb09a-4b4b-4ae6-92c4-7eda33b7a332" src="https://github.com/user-attachments/assets/c928be99-1545-4bd9-8a83-b3aebccbfc61" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

