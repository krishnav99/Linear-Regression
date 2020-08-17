import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#Reading the data
data = pd.read_csv("student_scores .csv")

#Displaying the dataset
print("Displaying the data\n",data)
data = np.array(data)
plt.scatter(data[:,0],data[:,1])
plt.xlabel("Hours studied")
plt.ylabel("Marks scored")
plt.show()

#inputing the dataset
inputs = data[:,0]
inputs = np.reshape(inputs, (-1,1))
outputs= data[:,1]

#splitting the dataset
train_input, test_input, train_output, test_output = train_test_split(inputs, outputs, test_size=0.20)

#compiling the model
model = LinearRegression()

#traing the model
model.fit(train_input,train_output)

#predicting using the model
prediction=model.predict(test_input)
print("\nTesting the model on the test data")
for i in range(len(test_output)):
               print("Hours studied:",test_input[i][0]," Marks predicted:%.2f"%prediction[i]," Target Marks:", test_output[i])

#evaluating the model
print("Mean absolute error: ",mean_absolute_error(test_output,prediction))

#predicting for the given data point
prediction = model.predict([[9.25]])
print("\nThe predicted score for a student studying 9.5 hours a day: %.2f"%prediction)

# Plotting the regression line
line = model.coef_*inputs+model.intercept_
plt.scatter(data[:,0],data[:,1])
plt.plot(inputs, line)
plt.xlabel("Hours studied")
plt.ylabel("Marks scored")
plt.show()
