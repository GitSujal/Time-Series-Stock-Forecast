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

'''The X-factors of the ARIMAX models are defined as follows.
Each factors are likely to affect the value of stock and were connsidered into account.
The numbers represents the corresponding date in date_delta form. 
The value 1 represents the book_closure started at date_delta 139 and 0 represents it is closed on 157 and respectively for other factors'''
data.loc[(data['date_delta']>=139), 'Book_Closure'] = 1;
data.loc[(data['date_delta']<157), 'Book_Closure'] = 0;
data.loc[(data['date_delta']>=256), 'Book_Closure'] = 1;
data.loc[(data['date_delta']<272), 'Book_Closure'] = 0;
data.loc[(data['date_delta']>=282), 'Listing'] = 1;
data.loc[(data['date_delta']<282), 'Listing'] = 0;
data.loc[(data['date_delta']>=287), 'New_Senior_Managment'] = 1;
data.loc[(data['date_delta']<287), 'New_Senior_Managment'] = 0;
data.loc[(data['date_delta']>=465), 'FPO'] = 1;
data.loc[(data['date_delta']<465), 'FPO'] = 0;


#Formatting the graph and labeling the axis
plt.figure(figsize=(15,5))
plt.plot(data.index,data['LTP'])
plt.ylabel('Arun Valley Stock Values')
plt.ylabel('Date')
plt.title("Arun Valley Stock prediction using ARIMAX model")
plt.show()

# For ARIMAX model
model = pf.ARIMAX(data=data, formula='LTP~1+Book_Closure+Listing+New_Senior_Managment+FPO', ar=4, ma=4, family=pf.Normal())
x = model.fit("MLE")
#Printing the summary of the fit
x.summary()

#Plot the fit line
model.plot_fit(figsize=(15,10))
#Predicting and plotting 20 values using past values
model.plot_predict_is(h=50, figsize=(15,5))


