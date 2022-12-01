from django.shortcuts import render, redirect
from django.http import HttpResponse,JsonResponse
from django.views.decorators.csrf import csrf_protect,csrf_exempt
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.contrib.auth.models import User
from django.contrib import messages
import pandas as pd
import numpy as np
import json
import os
import sys
# from datetime import datetime as dt

#Machine Learning Code Headers
from forecast.  First_phase_trial import arima,sarima,enhanced_arima_model,enhanced_auto_ml_model,auto_model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Create your views here.

# def dashboard(request):
#     context={}
#     return render(request,'dashboard.html',context)
@login_required(login_url="/logout/")
def upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
        input_1 = request.POST['column']
        input_2 = request.POST['date']
        input_3 = request.POST['interval']
        # print('*'*80)
        # print(file_path)
        # print(input_1)
        # print(input_2)
        # print(input_3)
        # print('*'*80)
        upload_data = file_path
        output_path = os.path.join(settings.MEDIA_ROOT,'Output')
        output_path = os.path.join(output_path,'forecast_test.csv')
        username=request.user.get_username().split('@')[0]
        print('*'*80)
        print(username)
        print('*'*80)
        # upload_data = r'D:\Local Disk D\Abinash\sales1.csv'
        try:
            df = pd.read_excel(upload_data,index_col=0,parse_dates=True)
            # df = pd.read_csv(df_excel,index_col=0,squeeze=True,parse_dates=True,infer_datetime_format=True,dayfirst=True)
        except Exception:
            
            df = pd.read_csv(upload_data,index_col=0,squeeze=True,parse_dates=True,infer_datetime_format=True,dayfirst=True)

        df = pd.DataFrame(df)
        # df = df.iloc[:,0]
        # df = pd.DataFrame(df)
        cols = str(input_1)
        df = df[cols]#Column will be selected here 
        df =pd.DataFrame(df) # make it to dataframe
        pred = df.copy() # not to be deleted
        pred = pred.reset_index()
        train = df[:int(len(df)*.80)]
        test = df[int(len(df)*.80):]

        # user will select whether it'll Month or day or week.
        # if user select Month it need to convert to 'M' 
        user_select = str(input_2)
        if user_select == 'Month':
            freq = 'M'
        elif user_select == 'Day':
            freq = 'D'
        elif user_select == 'week':
            freq='W'
        
        # freq = 'D' #for refference
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
            # print('*'*80)
            # print(input_3)
            # print(existing_value)
            # print(predict_value)
            # print(len(existing_value)+len(predict_value))
            # print('*'*80)
            # predict_value.extend(existing_value)
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
                        <td>'''+str(date[j])+'''</td>
                        <td class="text-right">'''+str(f_col[j])+'''</td>
                        <td class="text-right">'''+str(predict[j])+'''</td>
                    </tr>'''
            x+='</tbody>'
            context={'RMSE':round(RMSE,2),'MAPE':round(MAPE,2),'MAE':round(MAE,2),'module':best_module,'table':x,'weeks':weeks,'Channel':input_1,'Predict':existing_value,'M_Exist_Year':m_month_year_exist,'M_Exist_Value':m_month_value_exist,'M_Predict_Year':m_month_year_predict,'M_Predict_Value':m_month_value_predict,'M_List':m_list,'M_First':m_first,'M_Last':m_last,'user_select':user_select,'username':username}
        # Month Filter
        elif(user_select == "Month"):
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
            #print('='*80)
            # plot_table=df2[df2['Predicted Value'] > 0]
            # print('*'*80)
            # print(plot_table)
            weeks=""
            predict_value=""
            # print(weeks)
            # print(predict_value)
            # print('*'*80)
            date=month.index.strftime('%d-%m-%Y').tolist()
            f_col=month.iloc[:,0].tolist()
            predict=month.iloc[:,1].tolist()
            x='''<thead>
                    <tr class="bg-secondary text-white">
                        <th title="Field #1">Date</th>
                '''
            for i,col in zip(range(len(month.columns)),month.columns):
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
                        <td>'''+str(date[j])+'''</td>
                        <td class="text-right">'''+str(f_col[j])+'''</td>
                        <td class="text-right">'''+str(predict[j])+'''</td>
                    </tr>'''
            x+='</tbody>'
            context={'RMSE':round(RMSE,2),'MAPE':round(MAPE,2),'MAE':round(MAE,2),'module':best_module,'table':x,'weeks':weeks,'Channel':input_1,'Predict':predict_value,'M_Exist_Year':m_month_year_exist,'M_Exist_Value':m_month_value_exist,'M_Predict_Year':m_month_year_predict,'M_Predict_Value':m_month_value_predict,'M_List':m_list,'M_First':m_first,'M_Last':m_last,'user_select':user_select,'username':username}
        return render(request,'dashboard.html',context)
    else:
        context={}
        username = request.user.get_username()
        try:
            username = request.user.get_username().split('@')[0]
        except:
            username=username
        context={'username':username}
        return render(request,'upload.html',context)

@csrf_exempt
def edit_upload(request):
    context={}
    if request.method == 'POST':
    	uploaded_file = request.FILES['docs']
    	if uploaded_file:
            fs = FileSystemStorage()
            if os.path.exists(os.path.join(settings.MEDIA_ROOT, uploaded_file.name)):
                os.remove(os.path.join(settings.MEDIA_ROOT, uploaded_file.name))
            fs.save(uploaded_file.name, uploaded_file)
            path=os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
            try:
                df = pd.read_excel(path)
            except:
                df = pd.read_csv(path)
            columns=[]
            for col in df.columns:
                columns.append(col)
            context=json.dumps({'columns':columns})
            return HttpResponse(context)

def loginPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        user = authenticate(request, username=username, password=password)
        
        if user is not None:
            login(request,user)
            return redirect('upload')
        else:
            messages.error(request,'Invalid Credentials')
            return redirect('loginPage')
    else:
        return render(request,'login.html')

def logoutUser(request):
    request.session.clear()
    logout(request)
    return redirect('loginPage')

def rpass(request):
    if request.method == 'POST':
        password = request.POST.get('old_password')
        new_password = request.POST.get('new_password')
        confirm_password = request.POST.get('confirm_password')
        username = request.user.get_username()
        user = authenticate(request, username=username, password=password)

        if user is not None:
            if new_password == confirm_password:
                u = User.objects.get(username__exact=username)
                u.set_password(new_password)
                u.save()
                messages.success(request,'Password updated Successfully...')
                return redirect('loginPage')
            else:
                messages.error(request,'Mismatch in New Password and Confirm Password')
                return redirect('rpass')
        else:
            messages.error(request,'Old Password Mismatch')
            return redirect('rpass')

    return render(request,'change-password.html')