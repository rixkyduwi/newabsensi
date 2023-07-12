from django.urls import path
from django.contrib.auth import views as auth_views
from . import views

urlpatterns = [
	path('',views.index),
	path('home/',views.home,name="home"),
	path('absensi_siswa/',views.home, name="student"),
    
	path('register/', views.registerPage, name="register"),
	path('login/', views.loginPage, name="login"),   
	path('logout/', views.logoutUser,name="logout"),

	path('create_dataset', views.create_dataset,name="create_dataset"),
	path('trainer', views.trainer,name="trainer"),
	# path('detect', views.TrackImages,name="detect"),
    # path('track_images/', views.TrackImages, name='track_images'),
    path('video_feed/', views.video_feed, name='video_feed'),

	path('profile/<str:pk>/', views.profile,name="profile"),
	path('all_students/', views.all_students,name="all_students"),
    path('presensi/', views.present,name="presensi"),
	path('absent_students/', views.absent_students,name="absent_students"),
	path('delete/<str:pk>',views.deleteStudent,name='delete'),

	path('report/',views.report,name="report"),
	path('send_file/',views.send_file,name="send_file"),
	path('send/',views.send,name="send"),

	path('reset_password/',auth_views.PasswordResetView.as_view(),name="reset_password"),
    path('reset_password_sent/',auth_views.PasswordResetDoneView.as_view(),name="password_reset_done"),
    path('reset/<uidb64>/<token>',auth_views.PasswordResetConfirmView.as_view(),name="password_reset_confirm"),
    path('reset_password_complete/',auth_views.PasswordResetCompleteView.as_view(),name="password_reset_complete"),
]
