import datetime as dt
# import streamlit as st
import numpy as np
import pandas as pd
from ThymeBoost import ThymeBoost as tb
import pmdarima as pm
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error,r2_score,mean_absolute_percentage_error
from statsmodels.tsa.stattools import adfuller,acf, pacf
from math import sqrt
import base64
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
from statsmodels.tsa.stattools import kpss
from pmdarima import pipeline
from pmdarima import model_selection
from pmdarima import preprocessing as ppc
from pmdarima import arima
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
import statsmodels.api as sm
import itertools
from pmdarima.arima import AutoARIMA
from datetime import datetime
from math import sqrt
from pmdarima.arima import ndiffs
#----------------------------------------GETTING DATA------------------------------------------------
def get_df(data):
  custom_date_parser = lambda x: datetime.strptime(x, "%Y-%m-%d")
  extension = data.name.split('.')[1]
  if extension.upper() == 'CSV':
    df = pd.read_csv(data,index_col=0,squeeze=True,parse_dates=True,
                     date_parser=custom_date_parser,
                     infer_datetime_format=True,dayfirst=True)
  elif extension.upper() == 'XLSX':
    df = pd.read_excel(data, engine='openpyxl')
  elif extension.upper() == 'PICKLE':
    df = pd.read_pickle(data) 
  return df

#-----------------------------------MODEL1-------------------------------------------------------


def arima(df,train,test,freq,interval):
    n_diffs = ndiffs(train, max_d=10)
    model=auto_arima(train, start_p=1, start_q=1,
                      test='adf',
                      max_p=5, max_q=5,
                      m=12,             
                      d=None,          
                      seasonal=True,   
                      start_P=0, 
                      D=None, 
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True, 
                      information_criterion='aic',
                      stepwise=True,
          random_state=50,n_fits=50)
    prediction = pd.DataFrame(model.predict(n_periods = len(test)),index=test.index)
    prediction.columns = ['predicted_data']
    rmse= sqrt(mean_squared_error(test.iloc[:,0],prediction['predicted_data']))
    r2_scor =r2_score(test.iloc[:,0],prediction['predicted_data'])
    r2_scor =r2_scor*100
    mae =mean_absolute_error(test.iloc[:,0],prediction['predicted_data'])
    mape =mean_absolute_percentage_error(test.iloc[:,0],prediction['predicted_data'])
    mape = mape*100
    x =df.index[-1]
    rng = pd.date_range(x, periods=interval,freq=freq)
    pred = pd.DataFrame(model.predict(n_periods = interval),index=rng)
    pred.columns=['Predicted value']
    test_df = pd.merge(left=test,right=prediction,left_index=True,right_index=True)
    test_df = pd.DataFrame(test_df)
    concat_df = pd.concat([df,pred])
    concat_df = concat_df.fillna(0)
    concat_df = concat_df.astype(int)
    
    
    return rmse,mae,mape,pred,test_df,concat_df



#-----------------------------------MODEL2---------------------------------------------------------

def sarima(df,train,test,freq,interval):
    p = d = q = range(0,3) 
    pdq = list(itertools.product(p,d,q)) 
    p2 = d2 = q2 = range(0, 2)
    pdq2 = list(itertools.product(p2,d2,q2)) 
    s = 12 
    pdqs2 = [(c[0], c[1], c[2], s) for c in pdq2]
    combs = {}
    aics = []
    # Grid Search Continued
    for combination in pdq:
        for seasonal_combination in pdqs2:
            try:
                model = sm.tsa.statespace.SARIMAX(df, order=combination, seasonal_order=seasonal_combination,
                                                 enforce_stationarity=False,
                                                 enforce_invertibility=False)
                model = model.fit()
                combs.update({model.aic : [combination, seasonal_combination]})
                aics.append(model.aic)

            except:
                continue

    best_aic = min(aics)
    # Modeling and forcasting
    model = sm.tsa.statespace.SARIMAX(df, order=combs[best_aic][0], seasonal_order=combs[best_aic][1],
                                                 enforce_stationarity=False,
                                                 enforce_invertibility=False)
    sarima = model.fit()
    #model.forecast(len(test))
    pred = sarima.get_prediction(start=len(train), dynamic=False)
    pred_ci = pred.conf_int()
    y_pred = pred.predicted_mean
    y_pred = pd.DataFrame(y_pred)
    mse = np.square(np.subtract(test,y_pred)).mean()
    rmse = sqrt(mse)
    r2_scor =r2_score(test,y_pred)

    mae =mean_absolute_error(test,y_pred)
    mape =mean_absolute_percentage_error(test, y_pred)
    mape = mape*100
        #pred_uc = sarima.get_forecast(steps=12)
        #pred_ci = pred_uc.conf_int()
    x =df.index[-1]
    rng = pd.date_range(x, periods=interval,freq=freq)
    pred_uc = sarima.get_forecast(steps=interval)

    pred_ci = pred_uc.conf_int()
    forecast = pred_uc.predicted_mean
        #pred = pd.DataFrame(forecast,index=rng)
    pred = pd.DataFrame({  'Predicted Value': forecast})
    concat_df = pd.concat([df,pred])
    concat_df = concat_df.fillna(0)
    concat_df = concat_df.astype(int)
    
    return rmse,mae,mape,pred,y_pred,concat_df
#---------------------------------------MODEL 3---------------------------------------------------------
def auto_model(df,freq,interval):
    size = round(int(len(df)*.80))
    train, test = model_selection.train_test_split(df.iloc[:,0], train_size=size)
    
    # Let's create a pipeline with multiple stages... the Wineind dataset is
    # seasonal, so we'll include a FourierFeaturizer so we can fit it without
    # seasonality
    pipe = pipeline.Pipeline([
        ("fourier", ppc.FourierFeaturizer(m=12, k=4)),
        ("arima", AutoARIMA(stepwise=True, trace=1, error_action="ignore",
                                  seasonal=False,  # because we use Fourier
                                  suppress_warnings=True))
    ])
    
    pipe.fit(train)
    x =train.index[-1]
    rng = pd.date_range(x, periods=len(test), freq=freq)
    preds, conf_int = pipe.predict(n_periods=len(test), return_conf_int=True)
    
    auto_pred=pd.DataFrame(preds)
    m=pd.DataFrame(conf_int)
    auto_pred['Upper Bound'] = m.iloc[:,1]
    auto_pred['Lower Bound'] = m.iloc[:,0]
    pred = pd.DataFrame({ 'Date': rng, 'Forecast value': preds})
    pred = pd.DataFrame(pred)
    pred = pred.set_index('Date')
    rmse = sqrt(mean_squared_error(test,pred['Forecast value']))
    mae =mean_absolute_error(test, pred['Forecast value'])
    mape =mean_absolute_percentage_error(test, pred['Forecast value'])
    mape = mape*100
    pipe_model = pipeline.Pipeline([
        ("fourier", ppc.FourierFeaturizer(m=12, k=4)),
        ("arima", AutoARIMA(stepwise=True, trace=1, error_action="ignore",
                                  seasonal=False,  # because we use Fourier
                                  suppress_warnings=True))
    ])
    
    pipe_model.fit(df)
    x1 =df.index[-1]
    rng1 = pd.date_range(x1, periods=interval, freq=freq)
    preds1, conf_int1 = pipe_model.predict(n_periods=interval, return_conf_int=True)
    
    auto_pred1=pd.DataFrame(preds1)
    m1=pd.DataFrame(conf_int1)
    auto_pred1['Upper Bound'] = m1.iloc[:,1]
    auto_pred1['Lower Bound'] = m1.iloc[:,0]
    pred1 = pd.DataFrame({  'Date': rng1,'Predicted value': preds1})
    pred1 = pd.DataFrame(pred1)
    pred1 = pred1.set_index('Date')
    concat_df = pd.concat([df,pred])
    concat_df = concat_df.fillna(0)
    concat_df = concat_df.astype(int)
    
    return rmse,mae,mape,pred1,pred,test,concat_df



#---------------------------------------MODEL 4----------------------------------------------------------
def enhanced_auto_ml_model(df,freq,interval):
    
    if df is not None:
        df.columns=['ds','y']
        df['ds'] = pd.to_datetime(df['ds'],errors='coerce') 
        
        
        
        max_date = df['ds'].max()
        #st.write(max_date)
    
    
    
    if df is not None:
        m = Prophet()
        m.fit(df)

    if df is not None:
        future = m.make_future_dataframe(periods=interval,freq =freq)
        
        forecast = m.predict(future)
        fcst = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        combined_df = df.join(fcst, how = 'outer', lsuffix='_left', rsuffix='_right')
        fcst_filtered =  fcst[fcst['ds'] > max_date]    
       
        metric_df = fcst.set_index('ds')[['yhat']].join(df.set_index('ds').y).reset_index()
        metric_df.dropna(inplace=True)
        mse = np.square(np.subtract(metric_df.y,metric_df.yhat)).mean()
        rmse = sqrt(mse)
        r2_scor =r2_score(metric_df.y, metric_df.yhat)
        
        mae =mean_absolute_error(metric_df.y, metric_df.yhat)
        mape =mean_absolute_percentage_error(metric_df.y, metric_df.yhat)
        mape = mape*100
        pred = pd.DataFrame(fcst_filtered)
        pred = pred.set_index('ds')
        pred.columns = ['Predicted value','Lower Bound','Upper Bound']
        # concat_df = pd.concat([df,pred])
        # concat_df = concat_df.fillna(0)
        # concat_df = concat_df.astype(int)
        
    return rmse,mae,mape,pred,metric_df,combined_df
#---------------------------------------MODEL 5-------------------------------------------------------
def enhanced_arima_model(df,train,test,freq,interval):
    boosted_model = tb.ThymeBoost(verbose=0)
    model = boosted_model.autofit(train.iloc[:,0],
                               seasonal_period=0)
    predicted_output = boosted_model.predict(model, forecast_horizon=len(train))
    mse = mean_squared_error(train.iloc[:,0], predicted_output['predictions'])
    rmse =sqrt(mse)
    r2_scor =r2_score(train.iloc[:,0], predicted_output['predictions'])
    
    mae =mean_absolute_error(train.iloc[:,0], predicted_output['predictions'])
    mape =mean_absolute_percentage_error(train.iloc[:,0], predicted_output['predictions'])
    mape = mape*100
    boosted_model1 = tb.ThymeBoost(verbose=0)
    model1 = boosted_model1.autofit(df.iloc[:,0],
                               seasonal_period=12)
    x =df.index[-1]
    rng = pd.date_range(x, periods=interval, freq=freq)    
    predicted_output1 = boosted_model1.predict(model1, forecast_horizon=len(rng))
    #predicted_output1.columns = ['Predicted Value']
    concat_df = pd.concat([df,predicted_output])
    concat_df = concat_df.fillna(0)
    concat_df = concat_df.astype(int)
    return rmse,mae,mape,predicted_output1
#---------------------------------------DOWNLOAD THE FILE----------------------------------------

    
#---------------------------------------MAIN BLOCK-------------------------------------------------------  
# def main():
#     st.header('Upload the data with date column')
#     data = st.file_uploader("Upload file", type=['csv' ,'xlsx','pickle'])
#     if not data:
#       st.write("Upload a .csv or .xlsx file to get started")
#       return
#     df =get_df(data)
#     #pro = get_df1(data)
#     df =pd.DataFrame(df)
#     cols = st.selectbox(
#        'Please select a column',df.columns.tolist())
#     df = df[cols]
#     #pro = pro[cols]
#     df =pd.DataFrame(df)
#     #pro = df.copy()
#     #df= pd.to_datetime(df.index,infer_datetime_format=True,format='%Y-%m-%d',exact=True)
#     pred = df.copy()
#     pred = pred.reset_index()
#     #pred= pd.to_datetime(pred.iloc[:,0],infer_datetime_format=True,format='%Y-%m-%d',exact=True)
#     train = df[:int(len(df)*.75)]
#     test = df[int(len(df)*.75):]
#     model1 = arima(df,train,test)
#     model2 =sarima(df)
#     model3 =enhanced_auto_ml_model(pred)
#     model5=enhanced_arima_model(df, train,test)
#     model4 = auto_model(df)
#     my_dict ={'RMSE':[model1[0],model2[0],model3[0],model4[0],model5[0]],
#               #'R2_SCORE':[model1[1],model2[1],model3[1],model4[1],model5[1]],
#               'MAE':[model1[1],model2[1],model3[1],model4[1],model5[1]],
#               'MAPE':[model1[2],model2[2],model3[2],model4[2],model5[2]]
#              }
#     my_df=pd.DataFrame(my_dict,index=['ARIMA','SARIMA','ENHANCED AUTO ML MODEL','ENHANCED ARIMA MODEL','ENHANCED SARIMA MODEL'])
#     st.subheader('EVALUATION METRICS')
#     st.table(my_df)
#     min_val1 = float(my_df["RMSE"].min())
#     min_val = my_df['MAPE'].min()
#     min_val = float(min_val)
    


#     if model5[2]<float(20):
#         st.write('ENHANCED ARIMA MODEL is the best model for the dataset ')
#         #csv_exp = model5[4].to_csv(index=True)
#         download(model3[3])
#         st.line_chart(model3[3].iloc[:,0])
#         st.balloons()
      
#     elif model3[2]<float(20):
#         st.write('ENHANCED AUTO ML MODEL is the best model for the dataset ')
#         download(model3[3])
        
#         st.line_chart(model3[3].iloc[:,0])
#         st.balloons()

#     elif model2[2]<float(20):
#         st.write('SARIMA MODEL is the best model for the dataset ')
#         download(model2[3])
#         st.line_chart(model2[3].iloc[:,1])
#         st.balloons()
#     elif model1[2]<float(20):
#         st.write('ARIMA MODEL is the best model for the dataset ')
#         download(model1[3])
#         st.line_chart(model1[3].iloc[:,0])
#         st.balloons()
        
#     else:
#         st.write('ARIMA MODEL WITH PIPELINE   is the best model for the dataset ')
#         download(model4[3].iloc[:,0]) 
#         st.line_chart(model4[3].iloc[:,0])
#         st.balloons()
    
# main()