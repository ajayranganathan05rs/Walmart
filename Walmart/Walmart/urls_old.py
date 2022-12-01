from django.contrib import admin
from django.conf.urls import url
from django.urls import path, include
from forecast import views
from django.conf import settings
from django.conf.urls.static import static
from django.views.static import serve

urlpatterns = [
    path('admin/', admin.site.urls),
    path('edit_upload/', views.edit_upload , name="edit_upload"),
    path('edit_upload_1/', views.edit_upload_1 , name="edit_upload_1"),
    path('download_avg_dist/', views.download_avg_dist , name="download_avg_dist"),
    path('show_table/', views.show_table , name="show_table"),
    path('show_table_month/', views.show_table_month , name="show_table_month"),
    path('', views.loginPage , name="loginPage"),
    path('logout/', views.logoutUser , name="logoutUser"),
    url(r'^reset_password', views.rpass , name="rpass"),
    # url(r'^wfm/dashboard/', views.dashboard, name="dashboard"),
    url(r'^wfm/upload', views.upload, name="upload")
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)