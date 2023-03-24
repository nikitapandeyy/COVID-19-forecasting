import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA

# Load COVID-19 data
url = 'https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv'
df = pd.read_csv(url, usecols=['date', 'new_cases'])
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
df.set_index('date', inplace=True)

# Split data into training and testing sets
train = df.loc['2020-01-01':'2021-12-31']
test = df.loc['2022-01-01':]

# Fit ARIMA model
model = ARIMA(train, order=(2,1,2))
model_fit = model.fit(disp=0)

# Make predictions
start_index = len(train)
end_index = len(train) + len(test) - 1
forecast = model_fit.predict(start=start_index, end=end_index, typ='levels')

# Plot results
plt.plot(train.index, train, label='Training')
plt.plot(test.index, test, label='Testing')
plt.plot(forecast.index, forecast, label='Forecast')
plt.legend()
plt.show()
