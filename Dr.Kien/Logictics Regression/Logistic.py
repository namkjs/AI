import math
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from numpy import loadtxt, where
from pylab import scatter, show, legend, xlabel, ylabel

df = pd.read_csv("data.csv", header=0)

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
X = df[["grade1", "grade2"]]
X = np.array(X)
X = min_max_scaler.fit_transform(X)
Y = df["label"]  
Y = np.array(Y)


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

# Logistic regression model
def Logistic_Regression(X, Y, alpha, theta, num_iters):
    m = len(Y) 

    for x in range(num_iters): 
        new_theta = Gradient_Descent(X, Y, theta, m, alpha)
        theta = new_theta
        if x % 100 == 0:
            Cost_Function(X, Y, theta, m)
            print("Theta:", theta)
            print("Cost:", Cost_Function(X, Y, theta, m))

    Declare_Winner(theta) 

def Sigmoid(z):
    G_of_Z = 1.0 / (1.0 + math.exp(-z))
    return G_of_Z

# Gradient descent function
def Gradient_Descent(X, Y, theta, m, alpha):
    new_theta = []
    constant = alpha/m
    for j in range(len(theta)):
        CFDerivative = Cost_Function_Derivative(X, Y, theta, j, m, alpha)
        new_theta_value = theta[j] - CFDerivative
        new_theta.append(new_theta_value)
    return new_theta

# Cost function
def Cost_Function(X, Y, theta, m):
    sumOfErrors = 0
    for i in range(m):
        xi = X[i]
        hi = Hypothesis(theta,xi)
        if Y[i] == 1:
            error = Y[i] * math.log(hi)
        elif Y[i] == 0:
            error = (1-Y[i]) * math.log(1-hi)
        sumOfErrors += error
    const = -1/m
    J = const * sumOfErrors
    print ("cost is ", J )
    return J

# Hypothesis function
def Hypothesis(theta, x):
    z = 0
    for i in range(len(theta)):
        z += x[i]*theta[i]
    return Sigmoid(z)

# Declare winner function