import matplotlib
import pandas as pd
import numpy as np
from pandas import read_csv
from matplotlib import pyplot
import matplotlib.pyplot as plt
import statsmodels.graphics.tsaplots
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import math
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
import datetime
from sklearn.metrics import r2_score
import statsmodels.tsa.stattools as sts
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

#   2012-03-12 14:30



def plot_reshape_data(dataframe):

    print("Initial dataframe columns:", dataframe.columns)
    
    # Convert 'timestamp' to datetime format
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"])
    
    # Create 'dateTime' column and set as index
    dataframe["dateTime"] = dataframe["timestamp"]
    dataframe.set_index("dateTime", inplace=True)
    
    # Extract 'Weekday' from 'timestamp'
    dataframe["Weekday"] = dataframe["timestamp"].dt.dayofweek

    # Sort the DataFrame by 'dateTime' index
    dataframe = dataframe.sort_index()

    return dataframe

def fit_arima_model(df_train):
    model = ARIMA(df_train, order=(5,1,10))  # Example parameters, adjust as needed
    model_fit = model.fit()
    return model_fit

def predict_with_arima(model_fit, df_test, df_train):
    start = len(df_train)
    end = start + len(df_test) - 1
    df_predict = model_fit.predict(start=start, end=end)
    return df_predict

def calculate_rmse(df_test, df_predict):
    if len(df_test) != len(df_predict):
        raise ValueError("Length of test data and predictions do not match.")
    rms = sqrt(mean_squared_error(df_test, df_predict))
    return rms

    
# def Model_parameter(dataframe):
    
#     #ARIMA regression model parameter setting
#     plot_acf(dataframe['national'], lags=1300, alpha=0.05)
#     pyplot.show()
#     plot_pacf(dataframe['national'], lags=15, alpha=0.05)
#     pyplot.show()
#     #The data is stationary
#     print(sts.adfuller(dataframe.national))

#     return 0




def plot_acf_pacf(dataframe):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot ACF
    plot_acf(dataframe['national'], lags=1300, alpha=0.05, ax=ax1)
    ax1.set_title('Autocorrelation Function (ACF)')

    # Plot PACF
    plot_pacf(dataframe['national'], lags=15, alpha=0.05, ax=ax2)
    ax2.set_title('Partial Autocorrelation Function (PACF)')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show plots
    plt.show()

    # Print stationarity test result
    adf_result = sts.adfuller(dataframe['national'])
    print(f'ADF Statistic: {adf_result[0]}')
    print(f'p-value: {adf_result[1]}')
    print(f'Critical Values: {adf_result[4]}')





def PSI_prediction(data_csv_Address, my_date):

    # Load the data 
    try:
        dataframe = pd.read_csv(data_csv_Address, encoding='ISO-8859-1')  # Try this first
    except Exception as e:
        print("ISO-8859-1 encoding failed, trying utf-16.")
        dataframe = pd.read_csv(data_csv_Address, encoding='utf-16')  # Try utf-16 if the first fails

    # Ensure the date provided is in datetime format
    my_date = pd.to_datetime(my_date)
    
    # Prepare the DataFrame
    dataframe = plot_reshape_data(dataframe)
    
    # Ensure that 'dateTime' is set as the index and has timezone info
    if dataframe.index.tzinfo is None:
        dataframe.index = dataframe.index.tz_localize('UTC').tz_convert('Asia/Singapore')

    # Convert my_date to the same timezone as the DataFrame index
    my_date = my_date.tz_localize('UTC').tz_convert('Asia/Singapore')

    # Print the DataFrame to debug
    print("DataFrame head:\n", dataframe.head())
    
    # Find the index position of the provided date or the closest match
    
    if my_date not in dataframe.index:
        closest_date = dataframe.index.get_indexer([my_date], method='nearest')[0]
        closest_datetime = dataframe.index[closest_date]
        print(f"Exact date not found. Using closest date: {closest_datetime}")
        index_position = closest_date
    else:
        index_position = dataframe.index.get_loc(my_date)

    
    # ------------------------------------------- Slice the DataFrame into training and testing sets


    
    # Define the train-test split
    train_num = int(dataframe.shape[0] * 0.8)
    df_train = dataframe['national'][:train_num]
    df_test = dataframe['national'][train_num:]

    print("df_train: ", df_train)

    # # Model_parameters: 
    # Model_parameter(dataframe) 
    plot_acf_pacf(dataframe)

    # Train the model
    model_fit = fit_arima_model(df_train)
    print(model_fit.summary())
    
    # Make predictions
    df_predict = predict_with_arima(model_fit, df_test, df_train)
    

    # Calculate RMSE
    try:
        rms = calculate_rmse(df_test, df_predict)
        print('RMSE = ' + str(rms))

    except Exception as e:
        print(f"An error occurred while calculating RMSE: {e}")


    
    return model_fit.predict(start=index_position, end=index_position)
    

#   2017-03-12 14:30


def main():
    my_string = input('Enter date In between 2016_2019 (yyyy-mm-dd hh:mm): ')
    
    try:
        my_date = datetime.datetime.strptime(my_string, "%Y-%m-%d %H:%M")
    except ValueError:
        print("Date format is incorrect. Please use 'yyyy-mm-dd hh:mm'.")
        return
    
    PSI_prediction('psi_df_2016_2019.csv', my_date)
    

if __name__ == "__main__":
    main()


#--------------------------------------------------------------------------------- crossfold validation 


# Rolling Forecast Origin Cross-Validation
# In this approach, you use a rolling window to train and test the model. For each fold, the training set is expanded and the model is tested on the next time period.

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

def rolling_forecast_cv(series, order, window_size, step_size=1):
    """Perform rolling forecast cross-validation for ARIMA model.

    Args:
        series (pd.Series): The time series data.
        order (tuple): The ARIMA model order (p, d, q).
        window_size (int): The size of the rolling window.
        step_size (int): The step size for rolling window.

    Returns:
        list: A list of RMSE values for each fold.
    """
    rmses = []
    n = len(series)
    
    for start in range(0, n - window_size, step_size):
        train_end = start + window_size
        train = series[:train_end]
        test = series[train_end:train_end + step_size]
        
        model = ARIMA(train, order=order)
        model_fit = model.fit()
        
        forecast = model_fit.forecast(steps=step_size)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        rmses.append(rmse)
    
    return rmses

"""#------------------------------------------------------------------------------------------------- Example usage



series = pd.read_csv('psi_df_2016_2019.csv', encoding='ISO-8859-1', parse_dates=['timestamp'], index_col='timestamp')['national']
order = (5, 1, 10)  # Example ARIMA order
window_size = 1000  # Example rolling window size

rmses = rolling_forecast_cv(series, order, window_size)
print(f'Average RMSE: {np.mean(rmses)}')"""



# Expanding Window Cross-Validation
# In this approach, you start with an initial training set and progressively expand it while testing on subsequent time periods.

def expanding_window_cv(series, order, initial_size, step_size=1):
    """Perform expanding window cross-validation for ARIMA model.

    Args:
        series (pd.Series): The time series data.
        order (tuple): The ARIMA model order (p, d, q).
        initial_size (int): The size of the initial training set.
        step_size (int): The step size for expanding window.

    Returns:
        list: A list of RMSE values for each fold.
    """
    rmses = []
    n = len(series)
    
    for start in range(initial_size, n, step_size):
        train = series[:start]
        test = series[start:start + step_size]
        
        model = ARIMA(train, order=order)
        model_fit = model.fit()
        
        forecast = model_fit.forecast(steps=step_size)
        rmse = np.sqrt(mean_squared_error(test, forecast))
        rmses.append(rmse)
    
    return rmses

"""#------------------------------------------------------------------------------------------------- Example usage

initial_size = 1000  # Example initial training size
rmses = expanding_window_cv(series, order, initial_size)
print(f'Average RMSE: {np.mean(rmses)}')"""


##############################################################################  Visualize RMSE Cross Fold validation

def plot_cv_results(rmses, title):
    """Plot cross-validation results.

    Args:
        rmses (list): List of RMSE values from cross-validation.
        title (str): Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(rmses, marker='o', linestyle='-', color='b')
    plt.xlabel('Fold')
    plt.ylabel('RMSE')
    plt.title(title)
    plt.grid(True)
    plt.show()

"""#---------------------------------------------------------------------------------------------------------Example usage
plot_cv_results(rmses, 'Rolling Forecast Cross-Validation RMSE')"""