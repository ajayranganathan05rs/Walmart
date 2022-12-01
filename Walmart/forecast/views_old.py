from django.shortcuts import render, redirect
from django.http import HttpResponse,JsonResponse,Http404
from django.views.decorators.csrf import csrf_protect,csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import pandas as pd
import numpy as np
import json
import os
import sys
# from datetime import datetime as dt

#Machine Learning Code Headers
from forecast.  First_phase_trial import arima,sarima,enhanced_arima_model,enhanced_auto_ml_model,auto_model
from forecast.data_distribution import day_level,month1,interval
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create your views here.

# def dashboard(request):
#     context={}
#     return render(request,'dashboard.html',context)

def upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        uploaded_file1 = request.FILES['file1']
        uploaded_file2 = request.FILES['file2']
        uploaded_file3 = request.FILES['file3']
        file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
        file_path1 = os.path.join(settings.MEDIA_ROOT, uploaded_file1.name)
        file_path2 = os.path.join(settings.MEDIA_ROOT, uploaded_file2.name)
        file_path3 = os.path.join(settings.MEDIA_ROOT, uploaded_file3.name)
        # input_1 = request.POST['column']
        # input_2 = request.POST['date']
        input_3 = request.POST['interval']
        interval_90_days_input = file_path
        week_6_data_input = file_path1
        queue_offered_input = file_path2
        vendor_distribution_input = file_path3
        output_path = os.path.join(settings.MEDIA_ROOT,'Output')
        output_path = os.path.join(output_path,'forecast_test.csv')
        # upload_data = r'D:\Local Disk D\Abinash\sales1.csv'
        print('#'*80)
        # print('#'*80)
        # print('#'*80)
        try:
            interval_90_days = pd.read_excel(interval_90_days_input)
        except Exception:
            interval_90_days = pd.read_csv(interval_90_days_input)

        try:
            week_6_data = pd.read_csv(week_6_data_input)
        except:
            week_6_data = pd.read_excel(week_6_data_input)

        try:
            Queue_offered = pd.read_excel(queue_offered_input,index_col=0,parse_dates=True) #inputed by user
        except:
            Queue_offered = pd.read_csv(queue_offered_input,index_col=0,parse_dates=True)

        try:
            vendor=pd.read_csv(vendor_distribution_input)
        except:
            vendor=pd.read_excel(vendor_distribution_input)


        interval_90_days.drop(columns=interval_90_days.columns[0], 
                axis=1, 
                inplace=True)
        interval_90_days = interval_90_days.iloc[:,[0,7,12]]
                            #converting Datetime to date
        interval_90_days['Date'] = pd.to_datetime(interval_90_days['Date'] ,errors = 'coerce',format = '%Y-%m-%d').dt.strftime("%Y-%m-%d")
                            
        interval_90_days = interval_90_days.pivot_table(index=['Date'],columns= 'Interval 15 Minutes', values='Queue Offered',aggfunc=sum)
        interval_90_days = interval_90_days.fillna(0)
        day_level_func = day_level(interval_90_days)

        #Queued Offered for Forecast

        # df=day_level_data[2]
        df = pd.DataFrame(Queue_offered)
        df=pd.DataFrame(df)
        df=df.iloc[:,[0]]
        df=pd.DataFrame(df)
        input_1='Total'
        pred = df.copy() # not to be deleted
        pred = pred.reset_index()
        train = df[:int(len(df)*.80)]
        test = df[int(len(df)*.80):]

        user_select = 'Day'
        freq = 'M' #for refference
        interval =int(input_3) #12 #user need to input
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

        best_module = ''
        RMSE = float(my_df["RMSE"].min())
        MAPE = float(my_df["MAPE"].min())
        MAE = float(my_df["MAE"].min()) # Mean Absolute Error
        min_val1 = float(my_df["RMSE"].min())
        min_val = my_df['MAPE'].min()
        min_val = float(min_val)
        min_val2 = float(my_df['MAE'].min())

        if model4[2]<=min_val and model4[0]<=min_val1 and model4[1]<=min_val2:
            best_module='ENHANCED Auto ML MODEL is the best model for the dataset'
            val = model4[3]
            val = val['Predicted value']
            concat_df = pd.concat([df,val])
            # concat_df.columns.values[1]='Predicted Value'
            # concat_df = concat_df.columns[['Date','']]
            concat_df = concat_df.fillna(0)
            concat_df = concat_df.astype(int)
            concat_json = concat_df.to_json(orient='table')
            df2 = pd.DataFrame(concat_df.astype(int))
            df2 = df2.set_axis([input_1,'Predicted Value'], axis=1, inplace=False)
            month = df2.copy()
            month = df2.resample('MS').sum()
            df2.to_csv(output_path)
            print(df2)
        elif model2[2]<=min_val and model2[0]<=min_val1 and model2[1]<=min_val2:
            
            best_module='Sarima MODEL is the best model for the dataset'
            
            concat_df = pd.concat([df,model2[3]])
            concat_df = concat_df.fillna(0)
            concat_df = concat_df.astype(int)
            concat_df = pd.DataFrame(concat_df)
            concat_df = concat_df.columns[['Date','Predicted Value']]
            concat_json = concat_df.to_json(orient='table')
            df2= pd.DataFrame(concat_df.astype(int))
            df2 = df2.set_axis([input_1,'Predicted Value'], axis=1, inplace=False)
            month = df2.copy()
            month = df2.resample('MS').sum()
            df2.to_csv(output_path)
            print(df2)
        elif model1[2]<=min_val and model1[0]<=min_val1 and model1[1]<=min_val2:
            best_module='Arima MODEL is the best model for the dataset'
            concat_df = pd.concat([df,model1[3]])
            concat_df.columns.values[1]='Predicted Value'
            concat_df = concat_df.fillna(0)
            concat_df = concat_df.astype(int)
            concat_json = concat_df.to_json(orient='table')
            df2= pd.DataFrame(concat_df.astype(int))
            df2 = df2.set_axis([input_1,'Predicted Value'], axis=1, inplace=False)
            month = df2.copy()
            month = df2.resample('MS').sum()
            df2.to_csv(output_path)
            print(df2)
        else:
            best_module='ARIMA MODEL WITH PIPELINE is the best model for the dataset'
            concat_df = pd.concat([df,model3[3]])
            concat_df.columns.values[1]='Predicted Value'
            concat_df = concat_df.fillna(0)
            concat_df = concat_df.astype(int)
            concat_json = concat_df.to_json(orient='table')
            df2= pd.DataFrame(concat_df.astype(int))
            df2 = df2.set_axis([input_1,'Predicted Value'], axis=1, inplace=False)
            month = df2.copy()
            month = df2.resample('MS').sum()
            df2.to_csv(output_path)
            print(df2)

        # Day Filter
        if(user_select == "Day"):
            m_list=month.index.strftime('%b-%Y').tolist()
            m_plot=month[month['Predicted Value']>0]
            m_plot_list=m_plot.index.strftime('%b-%Y').tolist()
            m_first=m_plot_list[0]
            m_last=m_plot_list[-1]
            m_exist=month[month.iloc[:,0] > 0]
            m_month_year_exist=m_exist.index.strftime('%b-%Y').tolist()
            m_month_value_exist=m_exist.iloc[:,0].tolist()
            m_month_year_predict=month.index.strftime('%b-%Y').tolist()
            m_month_value_predict=month['Predicted Value'].tolist()
            # Exsiting 4 Week
            plot_table_1=df2[df2[input_1]>0]
            existing_value=plot_table_1[input_1].tolist()
            if len(existing_value) > 28:
                existing_value=existing_value[-28:]
            else:
                existing_value=existing_value
            # Predicting Weeks
            plot_table=df2[df2['Predicted Value'] > 0]
            weeks=plot_table.index.day_name().tolist()
            predict_value=plot_table['Predicted Value'].tolist()
            if len(predict_value) > 42:
                predict_value=predict_value[:42]
            else:
                predict_value = predict_value
            existing_value.extend(predict_value)
            date=df2.index.strftime('%d-%m-%Y').tolist()
            f_col=df2.iloc[:,0].tolist()
            predict=df2.iloc[:,1].tolist()
            x='''<thead>
                    <tr class="bg-secondary text-white">
                        <th title="Field #1">Date</th>
                '''
            for i,col in zip(range(len(df2.columns)),df2.columns):
                i=2+i
                if i == 3 or col == 0:
                    x+='    <th class="text-right" title="Field #3">Predicted Value</th>'
                else:
                    x+='    <th class="text-right" title="Field #'+str(i)+'">'+str(col)+'</th>'
            x+='''
                    </tr>
                </thead>
                <tbody>'''
            for j in range(len(date)):
                x+='''<tr>
                        <td class="text-center sorting_disabled">'''+str(date[j])+'''</td>
                        <td class="text-right">'''+str(f_col[j])+'''</td>
                        <td class="text-right">'''+str(predict[j])+'''</td>
                    </tr>'''
            x+='</tbody>'
            month_level_data = model4[3]
            month_level_data= pd.DataFrame(month_level_data)
            month_level_data.reset_index(inplace=True)
            month_level_data['month'] = month_level_data['ds'].dt.month_name()
            print(interval_90_days)
            month_func = month1(interval_90_days,month_level_data)
            def interval(df):
                month_func = month1(interval_90_days,month_level_data)
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

            # 6 Week of Data
            interval_level_data=interval(week_6_data)

            # Vendor Distribution
            print('*'*80)
            print(vendor)
            print('*'*80) 
            v_forecast = interval_level_data[5]
            v_interval = interval_level_data[6]
            # print(v_forecast)
            # print(v_interval)


            for i in range(len(vendor)):
                d_list=[]
                for j in range(len(v_forecast)):
                    d_list.append((v_forecast['forecast'][j]*vendor['Dist'][i])/100)
                v_forecast[vendor._get_value(i,'Vendors')]=d_list
                vendor_name=vendor._get_value(i,'Vendors')
                forecast_data=v_forecast['forecast'].tolist()
                forecast_day=v_forecast['day'].tolist()
                forecast_date=v_forecast['future_Date'].tolist()
                data_list=[[forecast_date[i],forecast_day[i],forecast_data[i],d_list[i]]for i in range(len(d_list))]
                vendor_wise_percent=pd.DataFrame(data_list,columns=['future_Date','day','forecast',vendor_name])
                interval_merge = v_interval.merge(vendor_wise_percent, on=['day'], how='left')
                interval_merge['month'] = interval_merge['future_Date'].dt.month_name()
                interval_merge.set_index(['day','future_Date','month'],inplace=True)
                interval_merge.reset_index(inplace=True)
                interval_merge.iloc[:,3:len(interval_merge.columns)-1] = (interval_merge.iloc[:,3:len(interval_merge.columns)-1].multiply(interval_merge.fillna(0).iloc[:,-1],axis=0))

                interval_merge = interval_merge.sort_values(by="future_Date",ascending=True)
                interval_merge.drop('forecast',axis=1)
                interval_merge['forecast']=forecast_data
                print('*'*80)
                print(interval_merge)
                print('*'*80)
                output_path_1 = os.path.join(settings.MEDIA_ROOT,'Vendor_Distribution')
                if not os.path.isdir(output_path_1):
                    os.makedirs(output_path_1)
                Vendor_Output_Path=os.path.join(output_path_1,vendor_name+'.csv')
                interval_merge.to_csv(Vendor_Output_Path,index=0)
            vendor_drop_down=''
            for ven_col in vendor['Vendors'].tolist():
                Vendor_Output_Path_1=os.path.join(output_path_1,ven_col+'.csv')
                vendor_drop_down+="""<a class="dropdown-item" href="#" onclick="func_ven('"""+ven_col+"""')" id='"""+ven_col+"""' downloads>"""+ven_col+"""</a>"""
                # vendor_drop_down+="""<a class="dropdown-item" href="#" download='"""+Vendor_Output_Path_1+"""'>"""+ven_col+"""</a>"""
            context={'RMSE':round(RMSE,2),'MAPE':round(MAPE,2),'MAE':round(MAE,2),'module':best_module,'table':x,'vendor_dropdown':vendor_drop_down,'weeks':weeks,'Channel':input_1,'Predict':existing_value,'M_Exist_Year':m_month_year_exist,'M_Exist_Value':m_month_value_exist,'M_Predict_Year':m_month_year_predict,'M_Predict_Value':m_month_value_predict,'M_List':m_list,'M_First':m_first,'M_Last':m_last,'user_select':user_select}
        return render(request,'dashboard.html',context)
    else:
        return render(request,'upload.html')

@csrf_exempt
def edit_upload(request):
    context={}
    if request.method == 'POST':
        uploaded_file = request.FILES['docs']
        uploaded_file1 = request.FILES['docs1']
        uploaded_file2 = request.FILES['docs2']
        uploaded_file3 = request.FILES['docs3']
        if uploaded_file:
            fs = FileSystemStorage()
            #90 Days Interval
            if os.path.exists(os.path.join(settings.MEDIA_ROOT, uploaded_file.name)):
                os.remove(os.path.join(settings.MEDIA_ROOT, uploaded_file.name))
            fs.save(uploaded_file.name, uploaded_file)
            path=os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
            #6 Week of Data
            if os.path.exists(os.path.join(settings.MEDIA_ROOT, uploaded_file1.name)):
                os.remove(os.path.join(settings.MEDIA_ROOT, uploaded_file1.name))
            fs.save(uploaded_file1.name, uploaded_file1)
            path=os.path.join(settings.MEDIA_ROOT, uploaded_file1.name)
            #Queue Offered
            if os.path.exists(os.path.join(settings.MEDIA_ROOT, uploaded_file2.name)):
                os.remove(os.path.join(settings.MEDIA_ROOT, uploaded_file2.name))
            fs.save(uploaded_file2.name, uploaded_file2)
            path=os.path.join(settings.MEDIA_ROOT, uploaded_file2.name)
            #Vendor
            if os.path.exists(os.path.join(settings.MEDIA_ROOT, uploaded_file3.name)):
                os.remove(os.path.join(settings.MEDIA_ROOT, uploaded_file3.name))
            fs.save(uploaded_file3.name, uploaded_file3)
            path=os.path.join(settings.MEDIA_ROOT, uploaded_file3.name)
            # print(path)
            try:
                df = pd.read_excel(path)
            except:
                df = pd.read_csv(path)
            columns=[]
            for col in df.columns:
                columns.append(col)
            val=len(df)
            print(val)
            print(columns)
            context=json.dumps({'columns':columns,'Length':val})
            return HttpResponse(context)

@csrf_exempt
def show_table(request):
    # context={}
    if request.method == 'POST':
        file_name = request.POST['path']
        f_path=os.path.join(settings.MEDIA_ROOT, 'Vendor_Distribution')
        file_name_ext=file_name+'.csv'
        f_path=os.path.join(f_path, file_name_ext)
        f_table=pd.read_csv(f_path)
        f_table=f_table.round(decimals = 1)
        f_table.rename(columns = {'day':'Day','future_Date':'Future Date','month':'Month','forecast':'Forecast'}, inplace = True)
        print('*'*80)
        print(f_table)
        print('*'*80)
        y=''
        date_new=f_table['Future Date'].tolist()
        f_col=f_table['Forecast'].tolist()
        v_col=f_table[file_name].tolist()
        y+='''<thead>
            <tr class="bg-secondary text-white">
        '''
        for i,col in zip(range(len(f_table.columns)),f_table.columns):
            print(col)
            y+='    <th class="text-center">'+str(col)+'</th>'
        y+='''
                </tr>
            </thead>
            <tbody>'''
        for j in range(len(date_new)):
            y+='''<tr>'''
            for k in f_table.iloc[j].tolist():
                y+='''<td class="text-center">'''+str(k)+'''</td>'''
            y+='''</tr>'''
        y+='</tbody>'
        z=''
        z+='''<thead>
            <tr class="bg-secondary text-white">
                <th class="text-center sorting_disabled">Date</th>
                <th class="text-center sorting_disabled">Forecast</th>
                <th class="text-center sorting_disabled">'''+file_name+'''</th>
            </tr>
            </thead>
            <tbody>'''
        for j in range(len(date_new)):
            z+='''<tr>
                    <td class="text-center">'''+str(date_new[j])+'''</td>
                    <td class="text-center">'''+str(f_col[j])+'''</td>
                    <td class="text-center">'''+str(v_col[j])+'''</td>
                </tr>'''
        z+='</tbody>'
        context=json.dumps({'table':y,'table1':z}) #,'table1':z
        return HttpResponse(context)