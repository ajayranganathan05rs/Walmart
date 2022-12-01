# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 21:56:07 2022

@author: Abinash.m
"""

import pandas as pd
import numpy as np

def day_level(df):
    day = df.copy()
    day['Total'] = day.sum(axis=1)
    day.reset_index(inplace=True)
    day['Date'] = pd.to_datetime(day['Date'])
    day['wom'] = day['Date'].apply(lambda d: (d.day-1) // 7+ 1)
    day['day'] = day['Date'].dt.day_name()
    day["wk&day"] = day["wom"].astype(str) + day["day"]
    day['month'] = day['Date'].dt.month_name()
    day['month_no'] = day['Date'].dt.month
    day.drop(day.iloc[:, 1:len(day.columns)-6], inplace=True, axis=1)
    day.sort_values(by='month_no',ascending=True)
    sum_month_wise = day.groupby(['month']).agg(
                sum_month_wise=('Total', 'sum')).reset_index()
    day = day.merge(sum_month_wise, on=['month'], how='left')
    day['call_percent'] = round((day['Total']/day['sum_month_wise'])*100,2)
    print(day)
    day = day.sort_values(by='Date',ascending=True)
    day_3_month = day.sort_values(by="Date",ascending=True).set_index("Date").last("3M")

    distribution_by_week_pivot   = day_3_month.pivot_table(index=['wk&day'],columns= 'month', values='call_percent', aggfunc='first')
    distribution_by_week_pivot['distribution'] = distribution_by_week_pivot.mean(axis=1)
    distribution_by_week_pivot = distribution_by_week_pivot
    day= day.merge(distribution_by_week_pivot, on=['wk&day'], how='left')
    day_copy = day.copy()
    day_copy.drop(['Date','Total','sum_month_wise','call_percent','month_no'],inplace=True,axis=1)
    day_copy = day_copy.groupby('wk&day').mean()
    day_copy =day_copy.fillna(0)
    day_copy['%dict'] = (day_copy['distribution']/sum(day_copy['distribution']))*100
    return day,sum_month_wise,day_3_month,distribution_by_week_pivot,day_copy

def month1(df,forecast):
    day_level_func = day_level(df)
    day_4_month = day_level_func[2]
    day_copy  = day_level_func[4]
    day_4_month.reset_index(inplace=True)
    forecast_month = forecast.copy()
    forecast_month = forecast_month.iloc[:,0:2]
    forecast_month.columns = ['future_Date','Predicted_value']
    forecast_month['future_Date'] = pd.to_datetime(forecast_month['future_Date'], format='%Y-%m-%d')
    forecast_month['month'] = forecast_month['future_Date'].dt.month_name() #forecatsed data
    forecast_month.drop(columns=forecast_month.columns[0], 
        axis=1, 
        inplace=True)
    month_day = forecast.copy()
    month_day['Start_date'] = month_day['ds'].apply(lambda x: x.strftime('%Y-%m-01'))
    month_day['Start_date'] = pd.to_datetime(month_day['Start_date'])
    m = month_day['Start_date'][0]
    month1 = month_day.sort_values(by="Start_date",ascending=False)
    length =len(month1)
    n = month1['ds'][length-1]
    future_date = pd.date_range(m, n, freq="D")#extracting date from forecasted data
    future_date = pd.DataFrame(future_date)
    future_date.columns = ['future_Date']
    month_df = future_date.copy()
    month_df['wom'] = month_df['future_Date'].apply(lambda d: (d.day-1) // 7+ 1)
    month_df['day'] = month_df['future_Date'].dt.day_name()
    month_df["wk&day"] = month_df["wom"].astype(str) + month_df["day"]
    month_df['month'] = month_df['future_Date'].dt.month_name()
    select_distribution =(day_copy['%dict'])
    select_distribution = pd.DataFrame(select_distribution)
    month_merge = month_df.merge(select_distribution, on=['wk&day'], how='left')
    forecast_df = month_merge.merge(forecast_month, on=['month'], how='left')
    forecast_df['forecast'] = ((forecast_df['%dict']*forecast_df['Predicted_value'])/100)
    forecast_df = round(forecast_df,2)
    return forecast_month,month_df,month_merge,forecast_df
def interval(df):
    month_func = month1(df,forecast_val)
    forecast_df = month_func[4]
    interval = df.copy()
    interval = interval.tail(42)
    interval.reset_index(inplace=True)
    interval['Date'] = pd.to_datetime(interval['Date'])
    interval = interval.sort_values(by="Date",ascending=True).set_index("Date").last("3M")
    interval.reset_index(inplace=True)
    interval['day'] = interval['Date'].dt.day_name()
    interval_day = interval.groupby('day').mean()
    interval_day['Total'] = interval_day.sum(axis=1)
    interval_day = round(interval_day)
    a = interval_day
    forecast_df_1 = forecast_df[['future_Date','day','forecast']]
    vendor_forecast = forecast_df_1 
    interval_df = interval_day.iloc[:,0:len(interval_day.columns)-1].divide(interval_day.iloc[:,-1],axis=0)
    vendor_interval = interval_df
    #interval_day = round(interval_day*100,2)
    d=interval_day
    interval_merge = interval_df.merge(forecast_df_1, on=['day'], how='left')
    interval_merge['month'] = interval_merge['future_Date'].dt.month_name()
    interval_merge.set_index(['day','future_Date','month'],inplace=True)
    interval_merge.reset_index(inplace=True)
    interval_merge.iloc[:,3:len(interval_merge.columns)-1] = (interval_merge.iloc[:,3:len(interval_merge.columns)-1].multiply(interval_merge.fillna(0).iloc[:,-1],axis=0))

    interval_merge = interval_merge.sort_values(by="future_Date",ascending=True)
    interval_merge_sum =  interval_merge.groupby(['month']).agg(
               sum_month_wise=('forecast', 'sum')).reset_index()
    return  interval,interval_day,interval_merge,interval_merge_sum,interval_df,vendor_forecast,vendor_interval


# df = pd.read_excel('C:/Users/abinash.m/Desktop/forecasting/forecastingv3.xlsx') #inputed by user
# vendor = pd.read_excel('C:/Users/abinash.m/Desktop/forecasting/vendor.xlsx') #inputed by user
# forecast_val = pd.read_csv('C:/Users/abinash.m/Desktop/forecasting/Forecast_data.csv') 
# df = df.iloc[:,[6,7,12]]
#                     #converting Datetime to date
# df['Date'] = pd.to_datetime(df['Date'] ,errors = 'coerce',format = '%Y-%m-%d').dt.strftime("%Y-%m-%d")
                    
# df = df.pivot_table(index=['Date'],columns= 'Interval 15 Minutes', values='Queue Offered',aggfunc=sum)
# df = df.fillna(0)
# day_level_func = day_level(df)
# #month_func = month(df)
# month_func1 = month1(df,forecast_val)
# interval_func = interval(df)
# vendor_col = list(vendor.iloc[:,0])
# datafr = {}
# for i in range(0,len(vendor)):
#     vendor_name = vendor_col[i]
#     data_name = vendor_name
    
#     vendor_for = interval_func[5]
#     vendor_int = interval_func[6]
#     vendor_for[vendor_name] = (vendor_for['forecast']*vendor['Dist'][i])/100
#     vendor_interval_merge = vendor_int.merge(vendor_for, on=['day'], how='left')
#     vendor_interval_merge['month'] = vendor_interval_merge['future_Date'].dt.month_name()
#     vendor_interval_merge.set_index(['day','future_Date','month'],inplace=True)
#     vendor_interval_merge.reset_index(inplace=True)
#     vendor_interval_merge.iloc[:,3:len(vendor_interval_merge.columns)-1] = (vendor_interval_merge.iloc[:,3:len(vendor_interval_merge.columns)-1].multiply(vendor_interval_merge.fillna(0).iloc[:,-1],axis=0))
    
#     vendor_interval_merge = vendor_interval_merge.sort_values(by="future_Date",ascending=True)
#     vendor_interval_merge = round(vendor_interval_merge,2)
#     datafr[data_name] =pd.DataFrame(vendor_interval_merge)