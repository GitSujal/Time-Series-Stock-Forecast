# Time-Series-Stock-Forecast
This python program performs forecast for stock based on time series models. This allows interpolation of upto 50 future points based on different popular time series method like Arima, Arimax and DAR model. This project also includes a sample data file in CSV format for testing purpose.

#Installing 
This program requires following dependencies to run.
* numpy for mathematical calculations
* Pandas for opening CSV and data preprocessing
* Matplotlib for plotting graphs
* Pyflux for the time series forecasting methods


#Running the program
A sample CSV file named Arun_Valley_Export.csv is provided with the project. You can use that data to forecast using different method. Run your preferred method with their respective file name.
Example:
```
python arima_model.py
python arimax_model.py
dar_model.py
```
You will see the graph plot for previous data and forecast using specified time series method in the graph plot.
