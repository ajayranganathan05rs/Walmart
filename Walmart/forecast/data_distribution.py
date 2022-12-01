# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 21:56:07 2022

@author: Abinash.m
"""

import pandas as pd
import numpy as np

def day_level(df,channel):
    day = df.copy()
    day['Date'] = pd.to_datetime(day['Date'])
    day['wom'] = day['Date'].apply(lambda d: (d.day-1) // 7+ 1)
    day['day'] = day['Date'].dt.day_name()
    day["wk&day"] = day["wom"].astype(str) + day["day"]
    day['month'] = day['Date'].dt.month_name()
    day['month_no'] = day['Date'].dt.month
    day.drop(day.iloc[:, 1:len(day.columns)-6], inplace=True, axis=1)
    day.sort_values(by='month_no',ascending=True)
    sum_month_wise = day.groupby(['month']).agg(
                            sum_month_wise=(channel, 'sum')).reset_index()
    day = day.merge(sum_month_wise, on=['month'], how='left')
    day['call_percent'] = round(((day[channel]/day['sum_month_wise'])*100),7)
    day = day.sort_values(by='Date',ascending=True)
    # day_4_month = day.sort_values(by="Date",ascending=True).set_index("Date").last(no_of_month)
    day_4_month = day.copy()
    distribution_by_week_pivot   = day_4_month.pivot_table(index=['wk&day'],columns= 'month', values='call_percent', aggfunc='first')
    distribution_by_week_pivot['distribution'] = distribution_by_week_pivot.median(axis=1)
    day= day.merge(distribution_by_week_pivot, on=['wk&day'], how='left')
    day_copy = day.copy()
    day_copy.drop(['Date',channel,'sum_month_wise','call_percent','month_no'],inplace=True,axis=1)
    day_copy = day_copy.groupby('wk&day').median()
    day_copy =day_copy.fillna(0)
    day_copy['%dict'] = round((day_copy['distribution']/sum(day_copy['distribution'])*100),7)                           
    day_copy1 = day_copy.copy()
    day_copy1.reset_index(inplace=True)
    df_samp = day_copy1['wk&day'].str.extract(r'^(\d+)(\D+)')
    week_5 = day_copy1.groupby(df_samp.loc[df_samp[0].isin(['1','2','3','4']), 1])['%dict'].median()
    day_copy1 = day_copy1.set_index('wk&day')
    day_copy1.loc['5Monday':'5Sunday', '%dict'] = week_5.add_prefix('5')
    day_copy1 = day_copy1.reset_index()
    day_copy1['Conversion'] = round((day_copy1['%dict']/sum(day_copy1['%dict']))*100,7)
    day_copy1 = day_copy1.set_index('wk&day')
    return day,sum_month_wise,day_4_month,distribution_by_week_pivot,day_copy,day_copy1,df

def future_month(forecast):
    day_data = day_level(df_day,channel)
    day_data = day_data[5]
    forecast['Date'] = pd.to_datetime(forecast['Date'], format='%d-%m-%Y')
    month_day = forecast.copy()
    month_day['Start_date'] = month_day['Date'].apply(lambda x: x.strftime('%Y-%m-01'))
    month_day['Start_date'] = pd.to_datetime(month_day['Start_date'])
    m = month_day['Start_date'][0]
    month1 = month_day.sort_values(by="Start_date",ascending=False)
    length =len(month1)
    n = month1['Date'][length-1]
    future_date = pd.date_range(m, n, freq="D")#extracting date from forecasted data
    future_date = pd.DataFrame(future_date)
    future_date.columns = ['Date']
    month_df = future_date.copy()
    month_df['wom'] = month_df['Date'].apply(lambda d: (d.day-1) // 7+ 1)
    month_df['day'] = month_df['Date'].dt.day_name()
    month_df["wk&day"] = month_df["wom"].astype(str) + month_df["day"]
    month_df['Month'] = month_df['Date'].dt.month_name()
    month_df['WK&Day'] = month_df['wk&day'].astype(str).str[0:4]
    month_wise_total = forecast.copy()
    month_wise_total.drop(['Date'],axis=1,inplace=True)
    # # select_distribution =(day_copy['%dict'])
    df_copy = day_data.copy()
    # df_copy.reset_index(inplace=True)
    # df_copy = df_copy.set_index('WK&Day')
    select_distribution =(df_copy['Conversion'])
    select_distribution = pd.DataFrame(select_distribution)
    month_merge = month_df.merge(select_distribution, on=['wk&day'], how='left')
    conversion_month_wise = month_merge.groupby(['Month']).agg(
                sum_month_wise=('Conversion', 'sum')).reset_index()
    month_merge = month_merge.merge(conversion_month_wise, on=['Month'], how='left')
    month_merge['Conversion%'] = (month_merge['Conversion']/month_merge['sum_month_wise'])*100
    forecast_df = month_merge
    forecast_df = forecast_df.merge(month_wise_total, on=['Month'], how='left')
    forecast_df['forecast'] = ((forecast_df['Conversion%']*forecast_df['Value'])/100)
    forecast_df = forecast_df
    return future_date,month_df,month_merge,month_wise_total,forecast_df,select_distribution
def interval(interval):
    forecast_df = future_month(forecast_val)
    forecast_df = forecast_df[4]
    interval_d = interval.copy()
    forecast_df_1 = forecast_df[['Date','Month','day','forecast']]
    interval_merge = interval_d.merge(forecast_df_1, on=['day'], how='left')
    interval_merge.set_index(['Date','Month','day'],inplace=True)
    interval_merge.reset_index(inplace=True)
    interval_merge.iloc[:,3:len(interval_merge.columns)-1] = (interval_merge.iloc[:,3:len(interval_merge.columns)-1].multiply(interval_merge.fillna(0).iloc[:,-1],axis=0))/100
    interval_merge = interval_merge.sort_values(by="Date",ascending=True)
    interval_merge_sum =  interval_merge.groupby(['Month']).agg(
                sum_month_wise=('forecast', 'sum')).reset_index()
    interval_merge_sum['sum_month_wise'] = interval_merge_sum['sum_month_wise'].astype(int)
    interval_merge_sum = round((interval_merge_sum),2)
    return forecast_df,forecast_df_1,interval_d,interval_merge,interval_merge_sum