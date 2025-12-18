# Implementation of Multivariate Linear Regression
## Aim
To write a python program to implement multivariate linear regression and predict the output.
## Equipment’s required:
1.	Hardware – PCs
2.	Anaconda – Python 3.7 Installation / Moodle-Code Runner
## Algorithm:
### Step1
import pandas

### Step2
Read the csv file

### Step3
Get the value of X and Y variable

### Step4
Create the linear regression model and fit

### Step5
Predict the CO2 emission of a car Where the Weight is 2300kg,and the volume is 1300cm cube

## Program:
```
 import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model,metrics
boston=datasets.load_diabetes(return_X_y=False)

#defining feature matrix(X) and response vector (y)
x=boston.data
y=boston.target
#splitting x and y into training and testing sets
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4,random_state=1)

#create linear regression object
reg=linear_model.LinearRegression()

#train the model using the training sets
reg.fit(x_train,y_train)

#regression coefficients
print("Coefficients",reg.coef_)

#variance score: 1means perfect prediction
print("Variance score: {}".format(reg.score(x_test,y_test)))

#plot for residual error
#setting plot style
plt.style.use("fivethirtyeight")

#plotting residual errors in training data
plt.scatter(reg.predict(x_train),reg.predict(x_train)-y_train,color='green',s=10,label="Train data")

#plotting residual errors in test data
plt.scatter(reg.predict(x_test),reg.predict(x_test)-y_test,color='blue',s=10,label="Test data")

#plotting line for zero residual error
plt.hlines(y=0,xmin=0,xmax=50,linewidth=2)

#plotting legend
plt.legend(loc='upper right')

#plot title
plt.title('Residual errors')

##method call for showing the plot
plt.show()

```
## Output:

 <img width="939" height="663" alt="Screenshot 2025-12-19 005540" src="https://github.com/user-attachments/assets/7994c8ed-da22-4563-b81e-68d0bc476daa" />


## Result
Thus the multivariate linear regression is implemented and predicted the output using python program.
