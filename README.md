# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import 'numpy' for numerical operations, 'pandas' for data manipulation, and various modules from 'sklearn' for machine learning tasks.

2.Fetch the California housing dataset using 'fetch_california_housing()' and convert it into a pandas DataFrame.

3.Define the feature set 'x' by dropping the 'AveOccup' and 'HousingPrice' columns from the DataFrame.

4.Define the target variable 'y' as a DataFrame containing 'AveOccup' and 'HousingPrice'.

5.Split the dataset into training and testing sets using 'train_test_split()'. The test size is set to 20% of the data.

6.Initialize 'StandardScaler' for both features and target variables.

7.Fit the scaler on the training data and transform both the training and testing sets for features ('x_train', 'x_test') and target variables ('y_train',' y_test').

8.Set the maximum number of iterations and the tolerance for stopping criteria in 'sgd'.

9.Use 'MultiOutputRegressor' to allow the 'SGDRegressor' to handle multiple target variables (in this case, 'AveOccup' and 'HousingPrice').

10.Fit the multi-output model on the scaled training data.

11.Use the trained model to predict the target variables for the scaled test set.

12.Inverse transform the predicted values and the actual test values to convert them back to their original scale.

13.Calculate the Mean Squared Error (MSE) between the actual and predicted values to assess the model's performance.

14.Print the first five predictions to see the output of the model

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by:SWETHA S 
RegisterNumber:212224040344


import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

dataset = fetch_california_housing()
df=pd.DataFrame(dataset.data,columns=dataset.feature_names)
df['HousingPrice']=dataset.target
print("Name: SWETHA S\nReg.no: 212224040344")
print(df.head())

x = df.drop(columns=['AveOccup', 'HousingPrice'])
y = df[['AveOccup', 'HousingPrice']]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

scaler_x = StandardScaler()
scaler_y = StandardScaler()

x_train = scaler_x.fit_transform(x_train)
x_test = scaler_x.transform(x_test)
y_train = scaler_y.fit_transform(y_train)
y_test = scaler_y.transform(y_test)

sgd = SGDRegressor (max_iter=1000, tol=1e-3)

multi_output_sgd=MultiOutputRegressor(sgd)

multi_output_sgd.fit(x_train,y_train)

y_pred = multi_output_sgd.predict(x_test)

y_pred = scaler_y.inverse_transform(y_pred)
y_test = scaler_y.inverse_transform(y_test)

mse=mean_squared_error(y_test,y_pred)
print("Mean Squared Error:",mse)

print("\nPredictions: \n", y_pred[:5])  
*/
```

## Output:
<img width="792" height="492" alt="Screenshot 2025-09-19 155010" src="https://github.com/user-attachments/assets/e3cb9df6-950b-4675-922f-50ec52007dde" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
