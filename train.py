import numpy as np 
import pandas as pd
from LinearRegression import LinearRegrssion
import matplotlib.pyplot as plt

data = pd.read_csv("train.csv")
data_test = pd.read_csv("test.csv")


#clear data from nan
data= data.dropna()
data_test= data_test.dropna()


X_train =data[['x']].values
y_train =data['y'].values
X_test =data_test[['x']].values
y_test =data_test['y'].values
print (X_train.shape)
print("test ",X_test.shape,y_test.shape)





model = LinearRegrssion()
model.fit(X_train,y_train)
X_test = np.array(X_test)
y_test = np.array(X_test) 
predictions = model.predict(X_test)

print("Weights:", model.weights)
print("Bias:", model.bias)


def mse(y_test,predictions):
    return np.mean((y_test-predictions)**2)
mse = mse(y_test,predictions)
print(mse)


y_pred_line = predictions
fig = plt.figure(figsize=(8,6))
cmap= plt.get_cmap('viridis')
plt.scatter(X_train,y_train,color = cmap(0.9) , s=10)
plt.scatter(X_test,y_test,color = cmap(0.5) ,  s=10)
plt.scatter(X_test,y_pred_line,color = "black" , linewidths=2,label ="Prediction")
plt.show()
