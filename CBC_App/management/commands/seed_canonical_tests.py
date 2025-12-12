# CBC_App/management/commands/seed_canonical_tests.py
from django.core.management.base import BaseCommand
from CBC_App.models import CanonicalTest

COMMON_TESTS = [
    {"code": "WBC", "display_name": "White Blood Cells", "typical_unit": "/mm3"},
    {"code": "RBC", "display_name": "Red Blood Cells", "typical_unit": "mil/mm3"},
    {"code": "HGB", "display_name": "Hemoglobin", "typical_unit": "g/dL"},
    {"code": "HCT", "display_name": "Hematocrit", "typical_unit": "%"},
    {"code": "MCV", "display_name": "MCV", "typical_unit": "fL"},
    {"code": "MCH", "display_name": "MCH", "typical_unit": "pg"},
    {"code": "MCHC", "display_name": "MCHC", "typical_unit": "g/dL"},
    {"code": "RDW-CV", "display_name": "RDW-CV", "typical_unit": "%"},
    {"code": "PLT", "display_name": "Platelets", "typical_unit": "/mm3"},
    {"code": "NEUT%", "display_name": "Neutrophils", "typical_unit": "%"},
    {"code": "LYMPH%", "display_name": "Lymphocytes", "typical_unit": "%"},
    {"code": "MONO%", "display_name": "Monocytes", "typical_unit": "%"},
    {"code": "EOS%", "display_name": "Eosinophils", "typical_unit": "%"},
    {"code": "BASO%", "display_name": "Basophils", "typical_unit": "%"},
    {"code": "BANDS%", "display_name": "Bands", "typical_unit": "%"},
    {"code": "Neutrophils", "display_name": "Neutrophils", "typical_unit": "%"},
]

class Command(BaseCommand):
    help = "Seed canonical CBC tests into CanonicalTest table"

    def handle(self, *args, **options):
        created = 0
        for t in COMMON_TESTS:
            obj, was_created = CanonicalTest.objects.get_or_create(
                code=t["code"],
                defaults={
                    "display_name": t["display_name"],
                    "typical_unit": t.get("typical_unit", ""),
                    "description": t.get("description", "")
                }
            )
            if was_created:
                created += 1
        self.stdout.write(self.style.SUCCESS(f"Seeded {created} canonical tests."))
