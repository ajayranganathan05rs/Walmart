"""Walmart URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.conf.urls import url
from django.urls import path, include
from partner_grade_book import views
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.loginPage , name="loginPage"),
    path('logout/', views.logoutUser , name="logoutUser"),
    url(r'^reset_password', views.rpass , name="rpass"),
    url(r'^partner_grade_book/home', views.home, name="home"),
    url(r'^partner_grade_book/attendance', views.attendance, name="attendance"),
    url(r'^partner_grade_book/observation', views.observation, name="observation"),
    url(r'^partner_grade_book/system_tracker', views.system_tracker, name="system_tracker"),
    url(r'^partner_grade_book/view_roster', views.view_roster, name="view_roster")
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)