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
plt.title("Arun Valley Stock prediction using DAR model")
plt.show()



#Using DAR model to forecast the time series
model = pf.DAR(data=data, ar=4, integ=0, target='LTP')
x = model.fit("MLE")
x.summary(transformed=False)
model.plot_fit(figsize=(15,10))

#Predicting 50 uture values using 100 past values
model.plot_predict(h=50, past_values=100, figsize=(15,5))