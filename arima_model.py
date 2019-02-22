
# Importing necessary dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyflux as pf


filename = "Arun_Valley_export.csv" #Name of file holding CSV data of stock values and Date


data = pd.read_csv(filename)
data['Date'] = pd.to_datetime(data['Date']) 

#Converting Data in mmddyyyy into date_delta integer
data['date_delta'] = (data['Date'] - data['Date'].min())  / np.timedelta64(1,'D')


#Delcaring index of the data
data.index = data['date_delta']


#Formatting the graph and labeling the axis
plt.figure(figsize=(15,5))
plt.plot(data.index,data['LTP'])
plt.ylabel('Arun Valley Stock Values')
plt.ylabel('Date')
plt.title("Arun Valley Stock prediction using ARIMA model")
plt.show()

# Using  Arima model
model = pf.ARIMA(data=data, ar=4, ma=4, target='LTP', family=pf.Normal())
x = model.fit("MLE")

#Printing the summary of the fit
x.summary()

#Predicting and plotting 20 values using past values
model.plot_predict(h=20,past_values=20,figsize=(15,5))

#Print the predicted values in terminal
print(model.predict(h=10,intervals=False))







