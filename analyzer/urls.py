from django.urls import path
from . import views

app_name = 'analyzer'

urlpatterns = [
    path('', views.index_view, name='index'),
    path('download-template/', views.download_template_view, name='download_template'),
    path('download-chart/<str:chart_name>/<str:ext>/', views.download_chart_view, name='download_chart'),
    path('chat/', views.chat_api_view, name='chat_api'),
]
