from django.shortcuts import render, redirect
from django.http import HttpResponse,JsonResponse,Http404
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
# from forecast.data_distribution import day_level,future_month,interval
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
        uploaded_file1 = request.FILES['file1']
        uploaded_file2 = request.FILES['file2']
        uploaded_file3 = request.FILES['file3']
        file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
        file_path1 = os.path.join(settings.MEDIA_ROOT, uploaded_file1.name)
        file_path2 = os.path.join(settings.MEDIA_ROOT, uploaded_file2.name)
        file_path3 = os.path.join(settings.MEDIA_ROOT, uploaded_file3.name)
        
        # Dyanmic data
        input_1 = request.POST['column']
        input_2 = int(request.POST['interval'])
        print(input_1)
        print(input_2)

        # Excluded
        # input_3 = request.POST['date']

        # static testing data
        # input_1 = 'Voice'
        # input_2 = 3

        # 90 days interval
        interval_90_days_input = file_path
        # 6 week interval
        week_6_data_input = file_path1
        # Forecast data -> uploaded directly by client
        queue_offered_input = file_path2
        # Vendor distribution -> uploaded directly by client
        vendor_distribution_input = file_path3
        # Output File
        output_path = os.path.join(settings.MEDIA_ROOT,'Output')
        output_path = os.path.join(output_path,'forecast_test.csv')
        username=request.user.get_username()
        # upload_data = r'D:\Local Disk D\Abinash\sales1.csv'
        print('#'*80)
        #  Upload File Check
        try:
            interval_90_days = pd.read_excel(interval_90_days_input)
        except Exception:
            interval_90_days = pd.read_csv(interval_90_days_input)

        try:
            week_6_data = pd.read_csv(week_6_data_input)
        except:
            week_6_data = pd.read_excel(week_6_data_input)

        try:
            Queue_offered = pd.read_excel(queue_offered_input)
        except:
            Queue_offered = pd.read_csv(queue_offered_input)

        try:
            vendor_distribution=pd.read_csv(vendor_distribution_input)
        except:
            vendor_distribution=pd.read_excel(vendor_distribution_input)

        #  Data Distribution Code Starts

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
            # month1 = month_day.sort_values(by="Start_date",ascending=False)
            # length =len(month1)
            # n = month1['Date'][length-1]
            n = month_day['Date'][1]
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

        #  Data Distribution Code Ends

        # Static Data
        # df_day = pd.read_excel('C:/Users/abinash.m/Desktop/forecastv2/DAY_LEVEL_DATA.xlsx') #inputed by user
        # forecast_val = pd.read_excel('C:/Users/abinash.m/Desktop/forecastv2/forecast_val_voice.xlsx')
        # interval_data= pd.read_excel(r'C:/Users/abinash.m/Desktop/forecastv2/Interval_data _voice.xlsx') #inputed by user

        # Dynamic Data
        df_day = interval_90_days #inputed by user
        forecast_val = Queue_offered
        interval_data= week_6_data

        # select_month = int(input('Please select no of month'))
        select_month = input_2
        if select_month==2:
            df_day = df_day.set_index('Date').last('2M')
        elif select_month==3:
            df_day = df_day.set_index('Date').last('3M')
        elif select_month==4:
            df_day = df_day.set_index('Date').last('4M')
        # channel = input('Please select a Channel')
        channel = input_1
        df_day =df_day[[channel]]
        df_day.reset_index(inplace=True)
        day_level_func = day_level(df_day,channel)
        month = future_month(forecast_val)
        interval_func = interval(interval_data)
        vendor1 = vendor_distribution #inputed by user
        vendor = vendor_distribution
        #month = future_month(df,forecast_val)
        #interval_func = interval(interval_data)
        vendor_col = list(vendor1.iloc[:,0])
        datafr = {}
        vendor_name = vendor_col[0]
        data_name = vendor_name
        vendor_for = interval_func[1]
        vendor_int = interval_func[2]
        vendor2= vendor1.copy()
        vendor2.drop(['Start','End'],inplace=True,axis=1)
        vendor2 = vendor2.T
        vendor2=vendor2.rename(columns=vendor2.iloc[0]).drop(vendor2.index[0])
        vendor2.reset_index(inplace=True)
        vendor2.rename(columns={'index':'Month'},inplace=True)
        vendor_for1=vendor_for.merge(vendor2,on=['Month'],how='left')
        vendor_for2 = vendor_for1.copy()
        vendor_for2.set_index(['Date','Month','day'],inplace=True)
        vendor_for2.iloc[:,1:len(vendor_for1.columns)-1] = (vendor_for2.iloc[:,1:len(vendor_for2.columns)].multiply(vendor_for2.fillna(0).iloc[:,0],axis=0))/100
        vendor_for2.reset_index(inplace=True)
        vendor_for2.set_index(['Date','Month','day','forecast'],inplace=True)
        vendor_col = list(vendor_for2.columns)

        for i in range(0,len(vendor_col)):
            vendor_name = vendor_col[i]
            data_name = vendor_name
            vendor_for3 = vendor_for2[vendor_name]
            vendor_for3 = pd.DataFrame(vendor_for3)
            vendor_for3.reset_index(inplace=True)
            vendor_interval_merge = vendor_int.merge(vendor_for3, on=['day'], how='left')
            vendor_interval_merge.set_index(['Date','Month','day','forecast'],inplace=True)
            
            vendor_interval_merge.iloc[:,0:len(vendor_interval_merge.columns)-1] = (vendor_interval_merge.iloc[:,0:len(vendor_interval_merge.columns)-1].multiply(vendor_interval_merge.fillna(0).iloc[:,-1],axis=0))/100
            vendor_interval_merge = vendor_interval_merge.sort_values(by="Date",ascending=True)
            vendor_interval_merge = round(vendor_interval_merge,2)
            datafr[data_name] =pd.DataFrame(vendor_interval_merge)
            
            

        hoops = {}
        for i in range(0,len(vendor_col)):
            data = list(datafr.values())[i]
            vendor_name = vendor_col[i]
            data_name = vendor_name
            data = pd.DataFrame(data)
            data.reset_index(inplace=True)
            dup = data[data.index.duplicated()]
            data = data[data[vendor_name]!=0]
            data.drop(['day','Month','forecast',vendor_name],axis=1,inplace=True)
            data1 = data.copy()
            data1['50Total'] = data1.sum(axis=1)
            print_data1 = data1.copy()
            data1 = data1.T
            data1 = data1.rename(columns=data1.iloc[0]).drop(data1.index[0])
            data1.reset_index(inplace=True)
            print_data2 = data1.copy()
            data1['To_char'] = data1['index'].astype(str).str[0:2]
            data1['To_char'] = data1['To_char'].replace('00','24')
            vendor1['start_to'] = vendor1['Start'].astype(str).str[0:2]
            vendor1['end_to'] = vendor1['End'].astype(str).str[0:2]
            start = vendor1['start_to'][i]
            end = vendor1['end_to'][i].replace('00','24')
            data2=data1[(data1['To_char'] >= start) & (data1['To_char'] <end)]
            print_data3 = data2.copy()
            data3 = data2.copy()
            data3.drop(['To_char'],axis=1,inplace=True)
            data3 =data3.T
            data3.reset_index(inplace=True)
            data3=data3.rename(columns=data3.iloc[0]).drop(data3.index[0])
            print_data4 = data3.copy()
            data1_1 =data1.copy()
            data1_1.drop(['To_char'],axis=1,inplace=True)
            data1_1=data1_1.T
            data1_1.reset_index(inplace=True)
            data1_1=data1_1.rename(columns=data1_1.iloc[0]).drop(data1_1.index[0])
            print_data5 = data1_1.copy()
            data1_2 = data1_1.copy()
            col = list(data1_1.columns)
            col = col[1:len(col)-1]
            for cols in col:
                data1_2[cols].values[:] = 0
            data1_2.update(data3)
            data1_2.set_index(['index','50Total'],inplace=True)
            data1_2['Total1']=data1_2.sum(axis=1)
            print_data6 = data1_2.copy()
            data4 = data1_2.iloc[:,0:len(data1_2.columns)-1].divide(data1_2.iloc[:,-1],axis=0)
            data4.reset_index(inplace=True)
            data4.set_index('index',inplace=True)
            data5 = round(data4.iloc[:,1:len(data4.columns)].multiply(data4.fillna(0).iloc[:,0],axis="index"),2)
            data5['total'] = data5.sum(axis=1)
            data5 = round(data5,2)
            data5.reset_index(inplace=True)
            data5.rename(columns={'index':'Date'},inplace=True)
            for c in data5.columns:
                if data5[c].dtype == np.object:
                    data5[c] = data5[c].astype(float).round(2).fillna(0.00)
            hoops[vendor_name] =pd.DataFrame(data5)

        print('*'*80)
        print(hoops)
        print('*'*80)
        vendor_dropdown=''
        for vendor_name in hoops:
            data=hoops[vendor_name]
            output_path = os.path.join(settings.MEDIA_ROOT,'Vendor_Distribution')
            output_path = os.path.join(output_path,vendor_name+'.csv')
            data.to_csv(output_path)
            vendor_dropdown+="""<a class="dropdown-item" href="#" onclick="func_ven('"""+vendor_name+"""')" id='"""+vendor_name+"""'>"""+vendor_name+"""</a>"""
        data=list(hoops.values())[0]
        month_list=data['Date'].dt.month_name().drop_duplicates().tolist()
        vendor_month_dropdown=''
        for month in month_list:
            vendor_month_dropdown+="""<a class="dropdown-item" href="#" onclick="func_ven_mon('"""+month+"""')" id='"""+month+"""'>"""+month+"""</a>"""
        # vendor_month_drop1=month_list[0]
        vendor_month_drop1="Select Month"
        vendor_drop1 = vendor_col[0]
        table=data.to_html(table_id='myTable',classes='mb-0 table table-bordered table-striped table-sm',index=False,justify="right",col_space=8)
        context={'vendor_dropdown':vendor_dropdown,'table':table,'vendor_drop1':vendor_drop1,'username':username,'vendor_month_dropdown':vendor_month_dropdown,'vendor_month_drop1':vendor_month_drop1}
        return render(request,'dashboard.html',context)
    else:
        context={}
        username=request.user.get_username()
        try:
            username=username.split('.')[0]+' '+username.split('.')[1].split('@')[0]
        except:
            username=username
        print(username)
        context={'username':username}
        return render(request,'upload.html',context)

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
def edit_upload_1(request):
    context={}
    if request.method == 'POST':
        uploaded_file = request.FILES['docs']
        if uploaded_file:
            fs = FileSystemStorage()
            #90 Days Interval
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
            val=len(df)
            print(val)
            print(columns)
            context=json.dumps({'columns':columns,'Length':val})
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


@csrf_exempt
def show_table(request):
    # context={}
    if request.method == 'POST':
        file_name = request.POST['path']
        f_path=os.path.join(settings.MEDIA_ROOT, 'Vendor_Distribution')
        vendor_list=[i.split('.csv')[0] for i in os.listdir(f_path) if '.csv' in i]
        file_name_ext=file_name+'.csv'
        f_path=os.path.join(f_path, file_name_ext)
        f_table=pd.read_csv(f_path,index_col=0)
        for c in f_table.columns:
            if f_table[c].dtype == np.float:
                f_table[c] = f_table[c].round(0).fillna(0).astype(int)
        print(file_name)
        month_list=f_table['Date'].astype('datetime64[ns]').dt.month_name().drop_duplicates().tolist()
        vendor_month_dropdown=''
        for month in month_list:
            vendor_month_dropdown+="""<a class="dropdown-item" href="#" onclick="func_ven_mon('"""+month+"""')" id='"""+month+"""'>"""+month+"""</a>"""
        vendor_month_drop1="Select Month"
        print(f_table)
        vendor_dropdown=''
        for vendor_name in vendor_list:
            vendor_dropdown+="""<a class="dropdown-item" href="#" onclick="func_ven('"""+vendor_name+"""')" id='"""+vendor_name+"""' downloads>"""+vendor_name+"""</a>"""
        y=f_table.to_html(table_id='myTable',classes='mb-0 table table-bordered table-striped table-sm',index=False,justify="right",col_space=8)
        drop_down='''<a class="nav-link dropdown-toggle" href="#" role="button" id="dropdownMenuLink" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">'''+file_name+'''</a> <div class="dropdown-menu" aria-labelledby="dropdownMenuLink">'''+vendor_dropdown+'''</div>'''
        drop_down1='''<a class="nav-link dropdown-toggle" href="#" role="button" id="dropdownMenuLink" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">'''+vendor_month_drop1+'''</a> <div class="dropdown-menu" aria-labelledby="dropdownMenuLink">'''+vendor_month_dropdown+'''</div>'''
        context=json.dumps({'table':y,'table1':y, 'drop_down':drop_down,'drop_down1':drop_down1}) #,'table1':z
        return HttpResponse(context)

@csrf_exempt
def show_table_month(request):
    # context={}
    if request.method == 'POST':
        file_name = request.POST['path']
        inp_month = request.POST['month']
        f_path=os.path.join(settings.MEDIA_ROOT, 'Vendor_Distribution')
        vendor_list=[i.split('.csv')[0] for i in os.listdir(f_path) if '.csv' in i]
        file_name_ext=file_name+'.csv'
        f_path=os.path.join(f_path, file_name_ext)
        f_table=pd.read_csv(f_path,index_col=0)
        for c in f_table.columns:
            if f_table[c].dtype == np.float:
                f_table[c] = f_table[c].round(0).fillna(0).astype(int)
        print(file_name)
        # Month Listing
        month_list=f_table['Date'].astype('datetime64[ns]').dt.month_name().drop_duplicates().tolist()
        vendor_month_dropdown=''
        for month in month_list:
            vendor_month_dropdown+="""<a class="dropdown-item" href="#" onclick="func_ven_mon('"""+month+"""')" id='"""+month+"""'>"""+month+"""</a>"""
        vendor_month_drop1=inp_month
        # Add Month column
        f_table["Month"] = f_table['Date'].astype('datetime64[ns]').dt.month_name()
        print(f_table)
        # Filtering month column
        f_table_month=f_table[f_table['Month']==inp_month]
        # Drop month column
        f_table_month.drop(['Month'],axis=1,inplace=True)
        print(f_table_month)
        # Vendor Listing
        vendor_dropdown=''
        for vendor_name in vendor_list:
            vendor_dropdown+="""<a class="dropdown-item" href="#" onclick="func_ven('"""+vendor_name+"""')" id='"""+vendor_name+"""' downloads>"""+vendor_name+"""</a>"""
        y=f_table_month.to_html(table_id='myTable',classes='mb-0 table table-bordered table-striped table-sm',index=False,justify="right",col_space=8)
        drop_down='''<a class="nav-link dropdown-toggle" href="#" role="button" id="dropdownMenuLink" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">'''+file_name+'''</a> <div class="dropdown-menu" aria-labelledby="dropdownMenuLink">'''+vendor_dropdown+'''</div>'''
        drop_down1='''<a class="nav-link dropdown-toggle" href="#" role="button" id="dropdownMenuLink" data-toggle="dropdown" aria-haspopup="true" aria-expanded="false">'''+vendor_month_drop1+'''</a> <div class="dropdown-menu" aria-labelledby="dropdownMenuLink">'''+vendor_month_dropdown+'''</div>'''
        context=json.dumps({'table':y,'table1':y, 'drop_down':drop_down,'drop_down1':drop_down1}) #,'table1':z
        return HttpResponse(context)