from django.contrib import admin

from django.contrib import admin
from django.contrib.auth.admin import UserAdmin as BaseUserAdmin
from .models import UserProfile, CanonicalTest, UploadDocument, TestResult,TestInterpretation, SharedReport

admin.site.register(UserProfile)
admin.site.register(TestInterpretation)
admin.site.register(SharedReport)
admin.site.register(CanonicalTest)
# admin.site.register(ReferenceRange)
admin.site.register(UploadDocument)
admin.site.register(TestResult)
# admin.site.register(Interpretation)
# admin.site.register(CorrectionLog)
