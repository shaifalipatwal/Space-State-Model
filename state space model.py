#Space State Model

import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# Load the dataset airline.csv

airline = pd.read_csv('C:/Users/SHAIFALI PATWAL/Desktop/Github Projects/airline.csv', parse_dates=['Month'], index_col='Month')

# Plotting the airline passanger dataset
plt.figure(figsize=(12, 5))
plt.plot(airline.index, airline['Passengers'], label='Passengers')
plt.xlabel('Date')
plt.ylabel('No. of Passengers')
plt.title('Monthly Airline Passengers')
plt.legend()
plt.show()


#%%
# Fitting the state space model with order=(1, 1, 1) and seasonal_order=(1, 1, 1, 12)
mod = sm.tsa.statespace.SARIMAX(airline['Passengers'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))
res = mod.fit(disp=False)
print(res.summary())

#%%

# Getting the residuals
residuals = res.resid

# Creating the PACF plot
sm.graphics.tsa.plot_pacf(residuals, lags=24)
plt.xlabel('Lag')
plt.ylabel('PACF')
plt.title('Partial Autocorrelation Function (PACF) of Residuals')
plt.show()

#%%
# Forecasting the future values
forecast = res.get_forecast(steps=24)
forecast_index = pd.date_range(start=airline.index[-1] + pd.DateOffset(1), periods=24, freq='M')
forecast_mean = forecast.predicted_mean
forecast_mean
forecast_ci = forecast.conf_int()
forecast_ci
#%%
# Plotting the forecast
plt.figure(figsize=(10, 6))
plt.plot(airline.index, airline['Passengers'], label='Observed Values')
plt.plot(forecast_index, forecast_mean, color='red', label='Forecasted values')
plt.fill_between(forecast_index, forecast_ci.iloc[:, 0], forecast_ci.iloc[:, 1], color='pink', alpha=0.3)
plt.xlabel('Date')
plt.ylabel('Number of Passengers')
plt.title('Forecasted Monthly Airline Passengers')
plt.legend()
plt.show()
