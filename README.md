# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import necessary libraries including NumPy, Pandas, and StandardScaler from sklearn.

2.Define a function linear_regression which takes input features (X1), target variable (y), learning rate (default value is 0.1), and number of iterations (default value is 1000). This function implements gradient descent to find the optimal parameters for the linear regression model.

3.Read the startup data from the CSV file into a DataFrame using Pandas.

4.Extract the feature matrix (X) and target variable (y) from the DataFrame. Convert X into a NumPy array and ensure it's of float data type. Also, scale both X and y using StandardScaler.

5.Call the linear_regression function with the scaled feature matrix (X1_Scaled) and target variable (Y1_Scaled) to obtain the optimal parameters (theta).

6.Create a new data point (new_data) and scale it using the same scaler used for the training data. Then, predict the profit for this new data point using the learned parameters (theta).

7.Inverse transform the predicted profit to get the original scale using the same scaler used for scaling the target variable.

8.Print the predicted profit.
## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: mohammed imthiyas.M
RegisterNumber:212222230083

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X = np.c_[np.ones(len(X1)),X1]
    theta = np.zeros(X.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions = (X).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta
data=pd.read_csv("C:/Users/admin/Downloads/AADITHYA/50_Startups.csv")
data.head()
X = (data.iloc[1:,:-2].values)
X1 =X.astype(float)
scaler = StandardScaler()
y = (data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled = scaler.fit_transform(X1)
Y1_Scaled = scaler.fit_transform(y)
print(X)
print(X1_Scaled)
theta=linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction= prediction.reshape(-1,1)
pre = scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value:{pre}")
*/
```

## Output:


![311636379-4aaf2aff-40c1-403e-89d8-89a91f5a5cf4](https://github.com/imthiyas19/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120353416/7ffe4f67-58de-4b2e-b4e3-1ca01880bc04)

![311636640-36bd2e0e-e7f0-4b27-b85e-848c05a7557b](https://github.com/imthiyas19/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120353416/6dcfecd1-163e-41a4-80ee-23f5083576ee)




![311636777-543ecbe1-b43e-4495-bd80-6af1d6f6b5da](https://github.com/imthiyas19/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120353416/5b0ef78e-c398-44d1-98f4-af9713f9acb8)


![311636842-39faa3d5-869b-41fc-a883-affdf8b31891](https://github.com/imthiyas19/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120353416/f831802e-a18b-4b15-a09d-8915b4a8bb3e)



![311636895-d49e0739-426b-44a8-9b69-cd4adb1bfaee](https://github.com/imthiyas19/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120353416/0f6e33ea-e0cc-4740-98ff-96bc1b739d5e)



![311636951-5eb6a538-c1fb-4ba9-ac56-41bbc8a10835](https://github.com/imthiyas19/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/120353416/89ec46c8-693d-446e-9341-a13011c6d674)






## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
