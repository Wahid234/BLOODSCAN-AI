# CBC_App/forms.py
from django import forms
from django.contrib.auth.forms import UserCreationForm, AuthenticationForm
from django.contrib.auth import get_user_model
from .models import UserProfile, UploadDocument

User = get_user_model()

GENDER_CHOICES = (
    ('', 'Select gender'),
    ('male', 'Male'),
    ('female', 'Female'),
)

ROLE_CHOICES = (
    ('patient', 'Patient'),
    ('doctor', 'Doctor'),
    ('admin', 'Admin'),
)

class SignUpForm(UserCreationForm):
    email = forms.EmailField(required=True)
    # profile fields:
    full_name = forms.CharField(required=False, max_length=200)
    role = forms.ChoiceField(choices=ROLE_CHOICES, initial='patient')
    mobile_number = forms.CharField(required=False, max_length=32)
    address = forms.CharField(widget=forms.Textarea, required=False)
    gender = forms.ChoiceField(choices=GENDER_CHOICES, required=False)
    birth_date = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}))
    profile_image = forms.ImageField(required=False)

    class Meta:
        model = User
        fields = ('username', 'email', 'password1', 'password2',
                  'full_name', 'role', 'mobile_number', 'address', 'gender', 'birth_date', 'profile_image')

    def save(self, commit=True):
        # Save User first
        user = super().save(commit=commit)
        # Save or update profile
        # Ensure profile exists (signals may create it; otherwise create/get)
        profile_data = {
            'role': self.cleaned_data.get('role'),
            'full_name': self.cleaned_data.get('full_name'),
            'mobile_number': self.cleaned_data.get('mobile_number'),
            'address': self.cleaned_data.get('address'),
            'gender': self.cleaned_data.get('gender'),
            'birth_date': self.cleaned_data.get('birth_date'),
        }
        profile, created = UserProfile.objects.get_or_create(user=user)
        for k, v in profile_data.items():
            setattr(profile, k, v)
        # handle profile image if provided
        image = self.cleaned_data.get('profile_image')
        if image:
            profile.profile_image = image
        profile.save()
        return user


class CustomAuthenticationForm(AuthenticationForm):
    # you can extend if you want additional validation
    pass


class UploadForm(forms.ModelForm):
    class Meta:
        model = UploadDocument
        fields = ('file',)
        widgets = {
            'file': forms.ClearableFileInput(attrs={'accept': '.png,.jpg,.jpeg,.pdf'})
        }

# CBC_App/forms.py  (أضف هذا)
from django import forms
from .models import TestResult

class TestResultVerifyForm(forms.ModelForm):
    class Meta:
        model = TestResult
        fields = ('test','value_raw','value_numeric','unit','verified')
        widgets = {
            'value_raw': forms.TextInput(attrs={'class':'form-control'}),
            'value_numeric': forms.NumberInput(attrs={'class':'form-control'}),
            'unit': forms.TextInput(attrs={'class':'form-control'}),
            'verified': forms.CheckboxInput(),
            'test': forms.TextInput(attrs={'class':'form-control'})  # or a select if CanonicalTest used
        }

class PdfUploadForm(forms.Form):
    file = forms.FileField(label="PDF report", help_text="Upload a PDF lab report (first page will be parsed).")

from django import forms

class ShareWithDoctorForm(forms.Form):
    doctor_id = forms.IntegerField(widget=forms.HiddenInput)  # doctor user id
    test_ids = forms.CharField(required=False, widget=forms.HiddenInput)  # comma separated ids
    document_id = forms.CharField(required=False, widget=forms.HiddenInput)
    message = forms.CharField(required=False, widget=forms.Textarea(attrs={"rows":3}), max_length=2000)

class DoctorRecommendationForm(forms.Form):
    shared_report_id = forms.CharField(widget=forms.HiddenInput)
    text = forms.CharField(widget=forms.Textarea(attrs={"rows":4}), max_length=4000)
    visible_to_patient = forms.BooleanField(required=False, initial=True)
