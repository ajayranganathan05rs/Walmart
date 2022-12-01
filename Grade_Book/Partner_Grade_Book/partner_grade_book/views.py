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

# Machine Learning Code Headers
import pandas as pd
import numpy as np

# Database MySQL
# import mysql.connector as sql

# Create your views here.

# def dashboard(request):
#     context={}
#     return render(request,'dashboard.html',context)
# @login_required(login_url="/logout/")
def home(request):
    context={}
    return render(request,'home.html',context)

# @login_required(login_url="/logout/")
def quick_attendance(request):
    context={}
    return render(request,'quick_attendance.html',context)

# @login_required(login_url="/logout/")
def observation(request):
    context={}
    return render(request,'observation.html',context)

# @login_required(login_url="/logout/")
def system_tracker(request):
    context={}
    return render(request,'system_tracker.html',context)

# @login_required(login_url="/logout/")
def view_roster(request):
    context={}
    return render(request,'view_roster.html',context)

@login_required(login_url="/logout/")
def upload(request):
    if request.method == 'POST':
        uploaded_file = request.FILES['file']
        file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
        upload_data = file_path
        username=request.user.get_username().split('@')[0]
        print('*'*80)
        print(username)
        print('*'*80)
        # upload_data = r'D:\Local Disk D\Abinash\sales1.csv'
        try:
            df = pd.read_excel(upload_data)
            # ,index_col=0,parse_dates=True
            # df = pd.read_csv(df_excel,index_col=0,squeeze=True,parse_dates=True,infer_datetime_format=True,dayfirst=True)
        except Exception:            
            df = pd.read_csv(upload_data)

        df = pd.DataFrame(df)
        table = df.to_html(table_id="myTable",classes="mb-0 table table-bordered table-striped table-sm",index=False,col_space=80,justify='right')
        # table = df.to_json(orient='records')
        # columns = df.columns.tolist()
        # print(table)
        # print(columns)
        # context={'table':table,'username':username,'columns':columns}
        context={'table':table,'username':username}
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
def upload_roaster(request):
    context={}
    if request.method == 'POST':
    	uploaded_file = request.FILES['docs']
    	if uploaded_file:
            fs = FileSystemStorage()
            if os.path.exists(os.path.join(settings.MEDIA_ROOT, uploaded_file.name)):
                os.remove(os.path.join(settings.MEDIA_ROOT, uploaded_file.name))
            fs.save(uploaded_file.name, uploaded_file)
            path=os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
            columns = ''
            try:
                df = pd.read_excel(path,sheet_name='LITMOS')
            except:
                try:
                    df = pd.read_csv(path)
                except:
                    columns = 'Error'
            if columns != 'Error':
                columns=[]
                for col in df.columns:
                    columns.append(col)
                # sql.connect(host='localhost',database='database',user='root',password='your password')
            context=json.dumps({'columns':columns})
            return HttpResponse(context)

def loginPage(request):
    return redirect('home')
    # if request.method == 'POST':
    #     username = request.POST.get('username')
    #     password = request.POST.get('password')

    #     user = authenticate(request, username=username, password=password)

    #     print('*'*80)
    #     print(user.get_username())
    #     print(type(user.get_username()))
    #     print('*'*80)
        
    #     if user is not None:
    #         login(request,user)
    #         # return redirect('upload')
    #         return redirect('home')
    #     else:
    #         messages.error(request,'Invalid Credentials')
    #         return redirect('loginPage')
    # else:
    #     return render(request,'login.html')

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