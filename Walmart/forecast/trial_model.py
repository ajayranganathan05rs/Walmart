from First_phase_trial import arima,sarima,enhanced_arima_model,enhanced_auto_ml_model,auto_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


#----------- Load the model-----------------------
'''
User need to input the data ,freq can be selected from 
drop down menu and interval can be inputed by the user.


'''
#-------Get the data---------------------
#data=

upload_data = 'sales1.csv'
try:
    df_excel = pd.read_excel(upload_data)
    df = pd.read_csv(df_excel,index_col=0,squeeze=True,parse_dates=True,
                     infer_datetime_format=True,dayfirst=True)
except Exception:
    
    df = pd.read_csv(upload_data,index_col=0,squeeze=True,parse_dates=True,
                 infer_datetime_format=True,dayfirst=True)

df = pd.DataFrame(df)
df = df.iloc[:,0]
df = pd.DataFrame(df)
#cols = <Once the user select a column it will reflect here>#df.columns.tolist())
#df = df[cols]#Column will be selected here 
df =pd.DataFrame(df) # make it to dataframe
pred = df.copy() # not to be deleted
pred = pred.reset_index()
train = df[:int(len(df)*.80)]
test = df[int(len(df)*.80):]
'''
user will select whether it'll Month or day or week.
if user select Month it need to convert to 'M' 
if user_select == 'Month':
    freq = 'M'
elif user_select == 'Day':
    freq = 'W'
elif user_select == 'week':
    freq='D'

'''
freq = 'D' #for refference
interval =12 #user need to input

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

RMSE = float(my_df["RMSE"].min())
MAPE = float(my_df["MAPE"].min())
MAE = float(my_df["MAE"].min()) # Mean Absolute Error
min_val1 = float(my_df["RMSE"].min())
min_val = my_df['MAPE'].min()
min_val = float(min_val)
#df2= pd.DataFrame(model1[3])
#df2.to_csv('forecast_test.csv')
if model4[2]<=min_val and model4[0]<=min_val1:
    print('ENHANCED Auto ML MODEL is the best model for the dataset ')
    # plot_model4 = model4[5]
    # plot_model4 = plot_model4.set_index(['ds_right'])
    # #plot_model4 = plot_model4[int(len(plot_model4)*.90):]
    # plt.figure(figsize=(12,5), dpi=100)
    # plt.plot(plot_model4['y'], color='blue', label='Actual Data')
    # plt.plot(plot_model4['yhat'], color='red', label='Forecast Value', alpha=0.6)
    # plt.title('Forecast vs Actuals')
    # plt.legend(loc='upper left', fontsize=8)
    # plt.show()
    df2= pd.DataFrame(model4[3].astype(int))
    df2.to_csv('forecast.csv')
elif model2[2]<=min_val and model2[0]<=min_val1:
    
    print('Sarima MODEL is the best model for the dataset ')
    #plot_model2 = model2[4]
    # plt.figure(figsize=(12,5), dpi=100)
    # plt.plot(test, color='blue', label='Actual')
    # plt.plot(model2[4], color='red', label='forecast', alpha=0.6)
    #plt.plot(n1, color='red', label='forecast') # forecated Data
    # plt.title('Forecast vs Actuals')
    # plt.legend(loc='upper left', fontsize=8)
    # plt.show()
    df2= pd.DataFrame(model2[3].astype(int))
    df2.to_csv('forecast_test.csv')
elif model1[2]<=min_val and model1[0]<=min_val1:
    print('Arima MODEL is the best model for the dataset ')
    # plot_model1 = model1[4]
    # plt.figure(figsize=(12,5), dpi=100)
    # plt.plot(plot_model1.iloc[:,0], color='blue', label='Actual')
    # plt.plot(plot_model1.iloc[:,1], color='red', label='forecast', alpha=0.6)
    # #plt.plot(model1[3], color='red', label='forecast') # forecated Data
    # plt.title('Forecast vs Actuals')
    # plt.legend(loc='upper left', fontsize=8)
    # plt.show()
    df2= pd.DataFrame(model1[3].astype(int))
    df2.to_csv('forecast_test.csv')
else:
    print('ARIMA MODEL WITH PIPELINE   is the best model for the dataset ')
    # plt.figure(figsize=(12,5), dpi=100)
    # plt.plot(test, color='blue', label='Actual')
    # plt.plot(model3[4], color='red', label='test_forecast', alpha=0.6)
    # #plt.plot(model3[3], color='red', label='forecast') # forecated Data
    # plt.title('Forecast vs Actuals')
    # plt.legend(loc='upper left', fontsize=8)
    # plt.show()
    df2= pd.DataFrame(model3[3].astype(int))
    df2.to_csv('forecast_test.csv')