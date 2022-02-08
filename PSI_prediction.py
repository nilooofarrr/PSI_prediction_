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
import datetime
from sklearn.metrics import r2_score
import statsmodels.tsa.stattools as sts
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

def plot_reshape_data(dataframe):

    # dataframe = pd.read_csv('psi_df_2016_2019.csv')
    plt.plot(range(dataframe.shape[0]), dataframe.national)#, tick_label=names)
    plt.show()
    
    dataframe["dateTime"] = pd.to_datetime(dataframe["timestamp"],format="%Y/%m/%d %H:%M:%S")
    dataframe["Weekday"] = dataframe["dateTime"].dt.dayofweek
    dataframe["Time"] = dataframe["dateTime"].dt.time
    
    #display 10 rows of data as a demo
    display(dataframe[0:10])
    
    return dataframe
    
def Model_parameter():
    
    #ARIMA regression model parameter setting
    plot_acf(dataframe['national'], lags=1300, alpha=0.05)
    pyplot.show()
    plot_pacf(dataframe['national'], lags=15, alpha=0.05)
    pyplot.show()
    #The data is stationary
    display(sts.adfuller(dataframe.national))


def predict_polution_psi(data_csv_Address, date_time):
    
    dataframe = pd.read_csv(data_csv_Address)
    dataframe=plot_reshape_data(dataframe)
    
    # n= dataframe.national.size
    n = dataframe.index[dataframe["dateTime"]==date_time]
    
    df_train = dataframe.national[0:n-1]
    #the p and q parameter obtain based on plot_reshape_data function acf and pacf
    model = ARIMA(df_train, order=(5,1,10))
    model_fit = model.fit()
    print(model_fit.summary())
    return model_fit.predict(start=n, end=n)
    



def model_psi_prediction():

    dataframe = pd.read_csv(data_csv_Address)
    dataframe=plot_reshape_data(dataframe)
    
    n= dataframe.national.size
    train_num = math.ceil(df.shape[0] * 0.8)
    df_train = df[0:train_num]
    test_num = df.shape[0] - train_num
    df_test = df[train_num + 1 : n]
    
    df_train = dataframe.national[0:n-1]
    #the p and q parameter obtain based on plot_reshape_data function acf and pacf
    
    model = ARIMA(df_train, order=(5,1,10))
    model_fit = model.fit()
    print(model_fit.summary())
    
    df_predict = model_fit.predict(start=train_num + 1, end=n)   
    
    #As we have a regression problem RMSE is more useful than precision
    rms = sqrt(mean_squared_error(df_test.national, df_predict.national))
    print('RMSE = '+str(rms))
    
    return rms



def main():
    
    my_string = str(input('Enter date(yyyy-mm-dd hh:mm): '))
    my_date = datetime.strptime(my_string, "%Y-%m-%d %H:%M")
    data_csv_Address ='psi_df_2016_2019.csv'
    predict_polution_psi(data_csv_Address, str(my_date) + ":00+08:00")
    rms = model_psi_prediction(data_csv_Address)
    print('RMSE = '+str(rms)) 

main()