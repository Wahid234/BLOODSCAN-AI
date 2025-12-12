# CBC_App/urls.py
from django.urls import path
from . import views

app_name = 'CBC_App'

urlpatterns = [
    path('', views.index, name='index'),
    path('signup/', views.signup_view, name='signup'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('upload/', views.upload_view, name='upload'),
    path('document/<uuid:document_id>/verify/', views.verify_document_view, name='verify_document'),
       path('manual-entry/', views.manual_entry_view, name='manual_entry'),
    path('manual-entry/process/', views.manual_entry_process, name='manual_entry_process'),  # optional JSON endpoint
    path('upload-pdf/', views.upload_pdf_report_view, name='upload_pdf'),
    path('api/doctors/', views.doctors_list_api, name='api_doctors_list'),
    path('share-with-doctor/', views.share_with_doctor, name='share_with_doctor'),
    path('doctor/inbox/', views.doctor_inbox_view, name='doctor_inbox'),
    path('my-shared/', views.my_shared_reports_view, name='my_shared_reports'),
    path('shared/<uuid:shared_id>/', views.shared_detail_view, name='shared_detail'),
    path('shared/add-recommendation/', views.add_recommendation, name='add_recommendation'),
    path('shared_accept/', views.shared_accept, name='shared_accept'),
    path('shared_reject/', views.shared_reject, name='shared_reject'),


]
