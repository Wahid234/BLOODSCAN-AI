from django.db import models

# CBC_App/models.py
import uuid
from django.contrib.auth.models import AbstractUser
from django.utils import timezone
from django.conf import settings


ROLE_CHOICES = (
    ('patient', 'Patient'),
    ('doctor', 'Doctor'),
    ('admin', 'Admin'),
)
from django.contrib.auth import get_user_model

User = get_user_model()
class UserProfile(models.Model):
    """
    Custom User extending Django's AbstractUser.
    Keep default id/username/email fields, add extra fields here.
    """
    # Note: do NOT redefine 'id' here if you want to keep Django default PK (integer).
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='patient')
    full_name = models.TextField()
    address = models.TextField(null=True, blank=True)
    mobile_number = models.TextField()
    gender = models.CharField(max_length=10, choices=[('male', 'Male'), ('female', 'Female')], blank=True, null=True)
    birth_date = models.DateField(blank=True, null=True)
    profile_image = models.ImageField(upload_to="profile_images/", blank=True, null=True)
    created_at = models.DateTimeField(auto_now_add=True) 
    updated_at = models.DateTimeField(auto_now=True)

    def is_patient(self):
        return self.role == 'patient'

    def is_doctor(self):
        return self.role == 'doctor'
    def __str__(self):
        return f"{self.user.username} profile"




class CanonicalTest(models.Model):
    """
    Normalized test types e.g., Hemoglobin (HGB), WBC, Platelets.
    """
    id = models.AutoField(primary_key=True)
    code = models.CharField(max_length=50, unique=True)
    display_name = models.CharField(max_length=150)
    typical_unit = models.CharField(max_length=50, null=True, blank=True)
    description = models.TextField(blank=True, null=True)
    

    def __str__(self):
        return f"{self.code} - {self.display_name}"


# class ReferenceRange(models.Model):
#     id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
#     test = models.ForeignKey(CanonicalTest, on_delete=models.CASCADE, related_name='reference_ranges')
#     min_value = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True)
#     max_value = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True)
#     unit = models.CharField(max_length=50, null=True, blank=True)
#     age_min = models.IntegerField(null=True, blank=True)
#     age_max = models.IntegerField(null=True, blank=True)
#     sex = models.CharField(max_length=10, null=True, blank=True)  # male/female/both
#     effective_from = models.DateField(null=True, blank=True)
#     effective_to = models.DateField(null=True, blank=True)
#     source = models.CharField(max_length=255, null=True, blank=True)


def upload_to_report(instance, filename):
    # example file path: uploads/userid/filename
    return f"uploads/{instance.user.id}/{filename}"

class UploadDocument(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='documents')
    file = models.FileField(upload_to=upload_to_report)
    filename = models.CharField(max_length=255, blank=True)
    mime_type = models.CharField(max_length=100, blank=True)
    upload_time = models.DateTimeField(default=timezone.now)
    reported_test_date = models.DateField(null=True, blank=True)
    status = models.CharField(max_length=32, default='pending')  # pending/processing/completed/failed
    ocr_confidence = models.DecimalField(max_digits=5, decimal_places=3, null=True, blank=True)
    raw_text = models.TextField(null=True, blank=True)
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Document {self.filename or self.id} ({self.user.username})"


class TestResult(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    document = models.ForeignKey(UploadDocument, on_delete=models.SET_NULL, null=True, blank=True, related_name='results')
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='test_results')
    test = models.ForeignKey(CanonicalTest, on_delete=models.SET_NULL, null=True, related_name='results')
    value_raw = models.CharField(max_length=100, blank=True)
    value_numeric = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True)
    unit = models.CharField(max_length=50, blank=True)
    value_date = models.DateField(null=True, blank=True)
    verified = models.BooleanField(default=False)
    verified_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True, related_name='verified_results')
    verified_at = models.DateTimeField(null=True, blank=True)
    confidence = models.DecimalField(max_digits=5, decimal_places=3, null=True, blank=True)
    # داخل class TestResult(models.Model):
    # أضف السطر التالي فوق value_raw أو بعده
    label_raw = models.CharField(max_length=255, blank=True, null=True, help_text="Original extracted label from OCR")

    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} - {self.test} : {self.value_raw}"


# class Interpretation(models.Model):
#     id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
#     result = models.OneToOneField(TestResult, on_delete=models.CASCADE, related_name='interpretation')
#     flag = models.CharField(max_length=20, null=True, blank=True)  # low/normal/high
#     text = models.TextField(null=True, blank=True)
#     generated_by = models.CharField(max_length=100, default='rules_engine_v1')
#     generated_at = models.DateTimeField(default=timezone.now)

#     def __str__(self):
#         return f"Interpretation for {self.result.id}"


# class CorrectionLog(models.Model):
#     id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
#     result = models.ForeignKey(TestResult, on_delete=models.CASCADE, related_name='corrections')
#     corrected_by = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='correction_logs')
#     previous_value_raw = models.CharField(max_length=100, blank=True)
#     previous_value_numeric = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True)
#     new_value_raw = models.CharField(max_length=100, blank=True)
#     new_value_numeric = models.DecimalField(max_digits=10, decimal_places=3, null=True, blank=True)
#     reason = models.TextField(null=True, blank=True)
#     corrected_at = models.DateTimeField(default=timezone.now)

#     def __str__(self):
#         return f"Correction {self.id} for {self.result.id}"
    
class TestInterpretation(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    test_result = models.ForeignKey(TestResult, on_delete=models.CASCADE, related_name='interpretations', null=True, blank=True)
    document = models.ForeignKey(UploadDocument, on_delete=models.CASCADE, related_name='interpretations', null=True, blank=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True)
    flag = models.CharField(max_length=20, null=True, blank=True)   # low/normal/high/unknown/error
    text = models.TextField(null=True, blank=True)                 # human readable interpretation
    meta = models.JSONField(null=True, blank=True)                 # store full interpreter payload if needed
    created_at = models.DateTimeField(default=timezone.now)
    class Meta:
        ordering = ['-created_at']



# --- new models for sharing & doctor responses ---

class SharedReport(models.Model):
    """
    Represents a sharing action where a patient shares one document (or multiple test results)
    with a doctor. ManyToMany to TestResult to allow sharing multiple tests in one share.
    """
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('viewed', 'Viewed'),
        ('responded', 'Responded'),
        ('archived', 'Archived'),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    sender = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='shared_reports_sent')
    doctor = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='shared_reports_received')
    document = models.ForeignKey('UploadDocument', on_delete=models.SET_NULL, null=True, blank=True, related_name='shared_reports')
    tests = models.ManyToManyField('TestResult', blank=True, related_name='shared_in_reports')
    message = models.TextField(null=True, blank=True, help_text="Optional message from patient")
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(default=timezone.now)
    viewed_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"SharedReport {self.id} from {self.sender} to {self.doctor}"

class DoctorRecommendation(models.Model):
    """
    Doctor's recommendation attached to a SharedReport.
    Allows multiple recommendations (history) if needed.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    shared_report = models.ForeignKey(SharedReport, on_delete=models.CASCADE, related_name='recommendations')
    doctor = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='recommendations')
    text = models.TextField()
    created_at = models.DateTimeField(default=timezone.now)
    visible_to_patient = models.BooleanField(default=True)

    class Meta:
        ordering = ['created_at']

    def __str__(self):
        return f"Recommendation {self.id} for {self.shared_report.id}"