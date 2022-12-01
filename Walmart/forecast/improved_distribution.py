# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:09:50 2022

@author: Abinash.m
"""

from First_phase_trial import arima,sarima,enhanced_arima_model,enhanced_auto_ml_model,auto_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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

    day = day.sort_values(by='Date',ascending=True)
    day_4_month = day.sort_values(by="Date",ascending=True).set_index("Date").last("3M")

    distribution_by_week_pivot   = day_4_month.pivot_table(index=['wk&day'],columns= 'month', values='call_percent', aggfunc='first')
    distribution_by_week_pivot['distribution'] = distribution_by_week_pivot.mean(axis=1)
    distribution_by_week_pivot = distribution_by_week_pivot
    day= day.merge(distribution_by_week_pivot, on=['wk&day'], how='left')
    day_copy = day.copy()
    day_copy.drop(['Date','Total','sum_month_wise','call_percent','month_no'],inplace=True,axis=1)
    day_copy = day_copy.groupby('wk&day').mean()
    day_copy =day_copy.fillna(0)
    day_copy['%dict'] = (day_copy['distribution']/sum(day_copy['distribution']))*100
    return day,sum_month_wise,day_4_month,distribution_by_week_pivot,day_copy
def month1(df,forecast):
    day_level_func = day_level(interval_90_days)
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
    select_distribution =(day_copy['interval_90_days'])
    select_distribution = pd.DataFrame(select_distribution)
    month_merge = month_df.merge(select_distribution, on=['wk&day'], how='left')
    forecast_df = month_merge.merge(forecast_month, on=['month'], how='left')
    forecast_df['forecast'] = ((forecast_df['distribution']*forecast_df['Predicted_value'])/100)
    forecast_df = round(forecast_df,2)
    return forecast_month,month_df,month_merge,forecast_df

interval_90_days = pd.read_csv('C:/Users/abinash.m/Desktop/forecasting/90_days_interval_data.csv') #inputed by user
interval_90_days.drop(columns=interval_90_days.columns[0], 
        axis=1, 
        inplace=True)
interval_90_days = interval_90_days.iloc[:,[0,7,12]]
                    #converting Datetime to date
interval_90_days['Date'] = pd.to_datetime(interval_90_days['Date'] ,errors = 'coerce',format = '%Y-%m-%d').dt.strftime("%Y-%m-%d")
                    
interval_90_days = interval_90_days.pivot_table(index=['Date'],columns= 'Interval 15 Minutes', values='Queue Offered',aggfunc=sum)
interval_90_days = interval_90_days.fillna(0)
day_level_func = day_level(interval_90_days)
Queue_offered = pd.read_excel('C:/Users/abinash.m/Desktop/forecasting/Queue_offered.xlsx',index_col=0,parse_dates=True) #inputed by user

df = pd.DataFrame(Queue_offered)
# input 1
#cols = 'Email'<Once the user select a column it will reflect here>#df.columns.tolist())
#df = df[cols]#Column will be selected here 
df =pd.DataFrame(df) # make it to dataframe
pred = df.copy() # not to be deleted
pred = pred.reset_index()
train = df[:int(len(df)*.70)]
test = df[int(len(df)*.70):]
#input 2
#user_select = 'Month'#user will select whether it'll Month or day or week.
'''
if user_select == 'Month':
    freq = 'M'
elif user_select == 'Day':
    freq = 'W'
elif user_select == 'week':
    freq='D'
    '''
#input 3
#interval =int() #user need to input
freq='M'
interval=6
model1 = arima(df, train, test, freq, interval)
model2 = sarima(df, train, test, freq, interval) 
model3 = auto_model(df, freq, interval)
model4 = enhanced_auto_ml_model(pred, freq, interval)
model5  = enhanced_arima_model(df, train, test, freq, interval)

my_dict ={'RMSE':[model1[0],model2[0],model3[0],model4[0],model5[0]],
           #'R2_SCORE':[model1[1],model2[1],model3[1],model4[1],model5[1]],
           'MAE':[model1[1],model2[1],model3[1],model4[1],model5[1]],
           'MAPE':[model1[2],model2[2],model3[2],model4[2],model5[2]]
          }
my_df=pd.DataFrame(my_dict,index=['ARIMA','SARIMA','ENHANCED Arima MODEL','ENHANCED AUTO ML MODEL','ENHANCED SARIMA MODEL'])
round_value= model1[3].astype(int)

min_val1 = float(my_df["RMSE"].min())
min_val = my_df['MAPE'].min()
min_val = float(min_val)
accuracy = 100-round(min_val,2)

min_val2 = float(my_df['MAE'].min())

if model4[2]<=min_val and model4[0]<=min_val1 and model4[1]<=min_val2:
    print('ENHANCED Auto ML MODEL is the best model for the dataset ')
    val = model4[3]
    val = val['Predicted value']
    concat_df = pd.concat([df,val])
    concat_df = concat_df.fillna(0)
    concat_df = concat_df.astype(int)
    concat_json = concat_df.to_json(orient='table')
    df2= pd.DataFrame(concat_df.astype(int))
    df2.to_csv('forecast.csv')
elif model2[2]<=min_val and model2[0]<=min_val1 and model2[1]<=min_val2:
    
    print('Sarima MODEL is the best model for the dataset ')
    
    concat_df = pd.concat([df,model2[3]])
    concat_df = concat_df.fillna(0)
    concat_df = concat_df.astype(int)
    concat_df = pd.DataFrame(concat_df)
    #concat_df = concat_df.columns[['Date','Predicted Value']]
    concat_json = concat_df.to_json(orient='table')
    df2= pd.DataFrame(concat_df.astype(int))
    df2.to_csv('forecast_test.csv')
elif model1[2]<=min_val and model1[0]<=min_val1 and model1[1]<=min_val2:
    print('Arima MODEL is the best model for the dataset ')
    concat_df = pd.concat([df,model1[3]])
    concat_df = concat_df.fillna(0)
    concat_df = concat_df.astype(int)
    concat_json = concat_df.to_json(orient='table')
    df2= pd.DataFrame(concat_df.astype(int))
    df2.to_csv('forecast_test.csv')
else:
    print('ARIMA MODEL WITH PIPELINE   is the best model for the dataset ')
    concat_df = pd.concat([df,model3[3]])
    concat_df = concat_df.fillna(0)
    concat_df = concat_df.astype(int)
    concat_json = concat_df.to_json(orient='table')
    df2= pd.DataFrame(concat_df.astype(int))
    df2.to_csv('forecast_test.csv')
forecast_val = model4[3]
forecast_val= pd.DataFrame(forecast_val)
forecast_val.reset_index(inplace=True)
forecast_val['month'] = forecast_val['ds'].dt.month_name()
month_func = month1(interval_90_days,forecast_val)
def interval(df):
    month_func = month1(interval_90_days,forecast_val)
    forecast_df = month_func[3]
    interval = interval_90_days.copy()
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
week_6_data = pd.read_csv('C:/Users/abinash.m/Desktop/forecasting/week_6_data.csv') #inputed by user

interval_func = interval(week_6_data)