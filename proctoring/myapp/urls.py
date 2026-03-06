from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),
    path("camera/",views.video_feed,name="video_feed")
]
