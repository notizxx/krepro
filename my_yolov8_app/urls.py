from django.contrib import admin
from django.urls import path, include
from detection_app import views
from detection_app.views import realtime_view, get_realtime_data
from detection_app.views import harga_buah_view
from detection_app.views import capture_image


urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.home, name='home'),  # Add the home view
    path('upload/', views.upload_image, name='upload_image'),
    path('detect/', views.detect_objects, name='detect_objects'),
    path('realtime/', views.detect_realtime, name='detect_realtime'),
    path('video_stream/', views.video_stream, name='video_stream'),
    path('about/', views.about, name='about'),
    path('intro/', views.intro, name='intro'),
    path('video_stream/', views.video_stream, name='video_stream'),
    path("api/realtime-data/", get_realtime_data, name="realtime_data"),  # API JSON
    path('harga-buah/', harga_buah_view, name='harga_buah'),  # Menampilkan halaman harga buah
    path('capture_image/', capture_image, name='capture_image'),
]
