# EX-3:Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe. 
2. Write a function computeCost to generate the cost function.  
3. Perform iterations og gradient steps with learning rate. 
4. Plot the Cost function using Gradient Descent and generate the required 
graph. 

## Program:
```
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: GOWTHAM S
RegisterNumber:2305002008 
```
```Python
# Making the imports 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
plt.rcParams['figure.figsize']=(12.0,9.0) 
data=pd.read_csv("profit.csv") 
X=data.iloc[:,0] 
Y=data.iloc[:,1] 
plt.scatter(X,Y) 
plt.show() 
#Building the model 
m=0 
c=0 
L=0.001 
epochs=5000 
n=float(len(X)) 
error=[] 
for i in range(epochs): 
    Y_pred=m*X+c 
    D_m=(-2/n)*sum(X*(Y-Y_pred)) 
    D_c=(-2/n)*sum(Y-Y_pred) 
    m=m-L*D_m 
    c=c-L*D_c 
    error.append(sum(Y-Y_pred)**2) 
print(m,c) 
type(error) 
print(len(error)) 
plt.plot(range(0,epochs),error) 
Y_pred=m*X+c 
plt.scatter(X,Y) 
plt.plot([min(X),max(X)],[min(Y_pred),max(Y_pred)],color="red") 
plt.show()
```

## Output:
![Screenshot 2024-05-19 192805](https://github.com/Ayvak16122005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147690197/94971945-2a8d-440f-9618-8e6bb89c126b)

![Screenshot 2024-05-19 192834](https://github.com/Ayvak16122005/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/147690197/5f0b0c97-ab33-43e4-9f1d-b900b48667f3)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

