# views.py
import tempfile
from django.shortcuts import render, redirect
from django.contrib.auth import login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .forms import SignUpForm, CustomAuthenticationForm, UploadForm,TestResultVerifyForm
from .models import UploadDocument
from django.shortcuts import render, redirect, get_object_or_404
from django.urls import reverse
from .forms import UploadForm
from .models import UploadDocument, TestResult, CanonicalTest, UserProfile
from .ocr import parse_ocr_text_advanced
from decimal import Decimal, InvalidOperation
from rapidfuzz import process as rf_process, fuzz as rf_fuzz

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages

from .forms import UploadForm
from .models import UploadDocument, TestResult, CanonicalTest
from .ocr import parse_ocr_text_advanced  # الدالة التي أنشأناها لمعالجة الصورة وOCR


@login_required
def index(request):
    documents = UploadDocument.objects.filter(user=request.user).order_by('-upload_time')[:5]
    q = SharedReport.objects.filter(sender=request.user).select_related('doctor','document')\
        .prefetch_related('recommendations','tests')[:5]

      # NEW: reports sent TO the user (if doctor)
    q_doctor = SharedReport.objects.filter(doctor=request.user)\
        .select_related('sender', 'document')\
        .prefetch_related('recommendations', 'tests')[:5]

    # load profile
    profile = None
    try:
        profile = request.user.userprofile
    except (AttributeError, UserProfile.DoesNotExist):
        profile = UserProfile.objects.filter(user=request.user).first()

    return render(request, 'index.html', {
        'documents': documents,
        'profile': profile,
        'shared_reports': q,              # reports sent BY the user
        'doctor_shared_reports': q_doctor # reports sent TO the doctor
    })

def signup_view(request):
    if request.method == 'POST':
        form = SignUpForm(request.POST, request.FILES)
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, "Account created successfully.")
            return redirect('CBC_App:index')
        else:
            messages.error(request, "Please correct the errors below.")
    else:
        form = SignUpForm()
    return render(request, 'signup.html', {'form': form})


def login_view(request):
    if request.method == 'POST':
        form = CustomAuthenticationForm(request, data=request.POST)
        if form.is_valid():
            user = form.get_user()
            login(request, user)
            return redirect('CBC_App:index')
        else:
            messages.error(request, "Invalid credentials.")
    else:
        form = CustomAuthenticationForm()
    return render(request, 'login.html', {'form': form})


def logout_view(request):
    logout(request)
    return redirect('CBC_App:index')
import pytesseract
from PIL import Image
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# in CBC_App/views.py (ضمن upload_view بعد doc.save() و doc.raw_text = ... )
from decimal import Decimal, InvalidOperation
from django.shortcuts import redirect
from .ocr import parse_ocr_text_advanced ,load_canonical_choices, fuzzy_map_label  # أو اسم الدالة لديك
from .models import CanonicalTest, TestResult
import re
from decimal import Decimal, InvalidOperation

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages

from rapidfuzz import process as rf_process, fuzz as rf_fuzz

from .forms import UploadForm
from .models import UploadDocument, TestResult, CanonicalTest
from .ocr import parse_ocr_text_advanced  # دالتك المتقدمة للـ OCR

# --- دالة لتنظيف النص وإستخلاص اسم الفحص فقط ---
UNIT_PATTERNS = [
    r"g\/dl", r"g/dl", r"fl", r"pg", r"/mm3", r"mm3", r"mil/mm3", r"/ul", r"µl", r"μl", r"%"
]
# نماذج لنطاقات مرجعية أو كلمات غير لازمة
REFERENCE_PATTERNS = [
    r"\b(reference|reference range|range|ref|result|result reference|reference range)\b",
    r"\b(test name|test)\b",
    r"\b(pathologist|comments|differential|count|hematology|blood complete picture|blood picture)\b",
]

def clean_label_text(label: str) -> str:
    """
    Clean OCR-extracted label string to get probable test name.
    - Lowercase
    - Remove parentheses content
    - Remove units, percentages, numbers, reference ranges
    - Remove common words (test/result/reference/headers)
    - Collapse whitespace and trim
    Returns cleaned label (possibly short) or empty string.
    """
    if not label:
        return ""

    s = label.strip().lower()

    # drop everything after colon (e.g., "Pathologist Comments : ...")
    s = re.split(r":", s)[0]

    # remove parentheses content
    s = re.sub(r"\([^)]*\)", " ", s)

    # remove common header words
    for pat in REFERENCE_PATTERNS:
        s = re.sub(pat, " ", s)

    # remove explicit units keywords
    for u in UNIT_PATTERNS:
        s = re.sub(r"\b" + u + r"\b", " ", s)

    # remove numeric ranges like "82 - 98" or "4,000 - 10,000" or "40-75"
    s = re.sub(r"\d{1,3}(?:[,\u00A0]\d{3})?(?:\.\d+)?\s*-\s*\d{1,3}(?:[,\u00A0]\d{3})?(?:\.\d+)?", " ", s)
    # remove standalone numbers (including decimals and commas)
    s = re.sub(r"[\d\u0660-\u0669\.,]+", " ", s)

    # remove slashes and extra punctuation
    s = re.sub(r"[/\\\-:%]", " ", s)

    # keep only letters and spaces
    s = re.sub(r"[^a-zA-Z\s]", " ", s)

    # collapse whitespace
    s = re.sub(r"\s+", " ", s).strip()

    return s

# --- دالة مطابقة محسّنة ---
def fuzzy_match_to_canonical(label_raw: str, canonical_qs, threshold: int = 90):
    """
    Returns (CanonicalTest instance or None, score, matched_name_or_none)
    canonical_qs: queryset of CanonicalTest
    threshold: percent threshold (0-100)
    Strategy:
      1) quick substring match (case-insensitive) on display_name or code
      2) fuzzy match on cleaned label against canonical display names
    """
    if not label_raw:
        return None, 0, None

    # prepare lists once (callers should cache if multiple calls)
    canonical_names = list(canonical_qs.values_list('display_name', flat=True))
    canonical_codes = list(canonical_qs.values_list('code', flat=True))

    # Quick substring match: if a canonical name appears in label_raw -> accept
    low = label_raw.lower()
    for name in canonical_names:
        if name.lower() in low:
            try:
                obj = canonical_qs.get(display_name=name)
                return obj, 100, name
            except CanonicalTest.DoesNotExist:
                continue
    # quick code check (e.g., "HGB" inside text)
    for code in canonical_codes:
        if code and code.lower() in low:
            try:
                obj = canonical_qs.get(code=code)
                return obj, 100, obj.display_name
            except CanonicalTest.DoesNotExist:
                continue

    # fallback: clean label and fuzzy match
    cleaned = clean_label_text(label_raw)
    if not cleaned:
        return None, 0, None

    # fuzzy match cleaned against canonical_names
    match = rf_process.extractOne(cleaned, canonical_names, scorer=rf_fuzz.token_sort_ratio)
    if match:
        matched_name, score, idx = match
        if score >= threshold:
            try:
                obj = canonical_qs.get(display_name=matched_name)
                return obj, score, matched_name
            except CanonicalTest.DoesNotExist:
                return None, score, matched_name
    return None, (match[1] if match else 0), (match[0] if match else None)


# --- upload_view المعدّلة ---
def upload_view(request):
    if request.method == 'POST':
        form = UploadForm(request.POST, request.FILES)
        if form.is_valid():
            doc = form.save(commit=False)
            doc.user = request.user
            doc.filename = doc.file.name
            doc.status = 'pending'
            doc.save()

            try:
                image_path = doc.file.path
                res = parse_ocr_text_advanced(image_path, lang='eng')
                doc.raw_text = res.get('raw_text', '')[:100000]
                doc.status = 'processed'
                doc.save()
            except Exception as e:
                messages.error(request, f"OCR processing failed: {e}")
                return render(request, 'upload.html', {'form': form})

            # تحميل canonical queryset مرة واحدة
            canonical_qs = CanonicalTest.objects.all()

            parsed_items = res.get('parsed', [])
            created_count = 0
            for item in parsed_items:
                # إن item قد يحتوي على 'label' المباشر و/أو 'cols'
                # استخرج label_raw الأولي (كما استخرجناه سابقًا)
                label_raw = (item.get('label') or "").strip()
                # لو label_raw فارغ، حاول استخراج من cols (انظر دالة derive سابقة)
                if not label_raw and item.get('cols'):
                    # pick left-most non-numeric column text
                    cols = item['cols']
                    for k in sorted(cols.keys()):
                        txt = cols.get(k, "").strip()
                        if txt:
                            label_raw = txt
                            break

                # value extraction (كما كنت تفعل)
                value_raw = str(item.get('value') or "").strip()
                unit = (item.get('unit') or "").strip()
                raw_obj = getattr(item.output, "raw_model_object", {}) or {}
                item.text_ar = (
                        getattr(item, "text_ar", None)
                        or getattr(getattr(item, "interpretation", None), "meta", {}).get("text_ar")
                        or raw_obj.get("text_ar")
                    )
                if not value_raw and item.get('cols'):
                    # search numeric in all cols (fallback)
                    for txt in item['cols'].values():
                        m = re.search(r"([0-9]{1,3}(?:[,\u00A0][0-9]{3})*(?:\.[0-9]+)?|\d+\.\d+|\d+)", txt)
                        if m:
                            value_raw = m.group(1).replace('\u00A0', '').replace(',', '')
                            break

                # clean numeric
                cleaned = value_raw.replace(',', '').replace('\u00A0', '').strip()
                if cleaned.endswith('%') and not unit:
                    unit = '%'
                    cleaned = cleaned.rstrip('%').strip()
                value_numeric = None
                if cleaned:
                    try:
                        value_numeric = Decimal(cleaned)
                    except (InvalidOperation, ValueError):
                        try:
                            value_numeric = Decimal(cleaned.replace(',', '.'))
                        except Exception:
                            value_numeric = None

                # fuzzy match improved (uses substring first, then cleaned fuzzy)
                canonical_obj, score, matched_name = fuzzy_match_to_canonical(label_raw, canonical_qs, threshold=90)

                # create TestResult with label_raw preserved
                TestResult.objects.create(
                    document=doc,
                    user=request.user,
                    test=canonical_obj,
                    label_raw=label_raw,
                    value_raw=value_raw,
                    value_numeric=value_numeric,
                    unit=unit,
                    verified=False
                )
                created_count += 1

            messages.success(request, f"File uploaded and OCR processed. {created_count} provisional results created.")
            return redirect('CBC_App:verify_document', document_id=doc.id)
        else:
            messages.error(request, "Upload form is invalid. Please correct the errors below.")
            return render(request, 'upload.html', {'form': form})
    else:
        form = UploadForm()
        return render(request, 'upload.html', {'form': form})
from django.forms import modelformset_factory
from django.contrib import messages


from .models import UploadDocument, TestResult

@login_required
def verify_document_view(request, document_id):
    doc = get_object_or_404(UploadDocument, id=document_id, user=request.user)
    TestResultFormSet = modelformset_factory(TestResult, form=TestResultVerifyForm, extra=0)
    qs = TestResult.objects.filter(document=doc).order_by('id')

    if request.method == 'POST':
        formset = TestResultFormSet(request.POST, queryset=qs)
        if formset.is_valid():
            for form in formset:
                obj = form.save(commit=False)
        
                obj.verified = True
                obj.save()
            messages.success(request, "Results verified and saved successfully.")
            return redirect('CBC_App:index')
    else:
        formset = TestResultFormSet(queryset=qs)

    # build parsed list from TestResult rows for display
    parsed = []
    for r in qs:
        parsed.append({
            'label_raw': r.label_raw or '(no label)',
            'mapped': r.test.display_name if r.test else '(unmatched)',
            'value': r.value_raw,
            'unit': r.unit or '',
        })

    context = {
        'doc': doc,
        'formset': formset,
        'parsed': parsed
    }
    return render(request, 'verify.html', context)


# CBC_App/views.py (أضف هذه الوظائف أو الصقها في أسفله)
import re
from decimal import Decimal, InvalidOperation

from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST

from .models import CanonicalTest, TestResult, UploadDocument, TestInterpretation
from .forms import UploadForm, TestResultVerifyForm  # ensure forms exist
from .utils.interpreter_service import interpret_rows

@login_required
def manual_entry_view(request):
    """
    Show manual entry form.
    """
    tests = CanonicalTest.objects.all().order_by('display_name')
    return render(request, 'manual_entry.html', {"tests": tests})

import json
@require_POST
@login_required
def manual_entry_process(request):
    """
    Process submitted manual rows, call interpreter (import/subprocess), save TestResult and TestInterpretation,
    normalize English/Arabic interpreter outputs (text_en/text_ar) and render results similar to upload_pdf_report_view.
    """
    # collect lists from POST -- same naming as inputs in template
    tests_sel = request.POST.getlist('test_select')
    labels = request.POST.getlist('label_raw')
    values = request.POST.getlist('value')
    units = request.POST.getlist('unit')
    ref_mins = request.POST.getlist('ref_min')
    ref_maxs = request.POST.getlist('ref_max')

    # build rows for interpreter
    rows = []
    for i in range(len(values)):
        # pick selected test if available
        test_obj = None
        test_name = ""
        if i < len(tests_sel) and tests_sel[i]:
            try:
                test_obj = CanonicalTest.objects.get(id=tests_sel[i])
                test_name = test_obj.display_name
            except (CanonicalTest.DoesNotExist, ValueError):
                test_name = labels[i] if i < len(labels) and labels[i] else "Unnamed test"
        else:
            test_name = labels[i] if i < len(labels) and labels[i] else "Unnamed test"

        raw_val = values[i] if i < len(values) else ""
        unit = units[i] if i < len(units) else ""

        # numeric conversion (Decimal) if possible
        num = None
        try:
            if raw_val and raw_val.strip() != "":
                num = Decimal(raw_val.replace(',', '').strip())
        except Exception:
            num = None

        # parse refs helper
        def parse_ref(x):
            try:
                return float(x.replace(',', '').strip()) if x and x.strip() != '' else None
            except Exception:
                return None

        rmin = parse_ref(ref_mins[i]) if i < len(ref_mins) else None
        rmax = parse_ref(ref_maxs[i]) if i < len(ref_maxs) else None

        rows.append({
            "test_code": test_obj.code if test_obj else "",
            "test_name": test_name,
            "value": float(num) if num is not None else (raw_val if raw_val != "" else None),
            "value_raw": raw_val,
            "unit": unit,
            "ref_min": rmin,
            "ref_max": rmax,
            "meta": {}  # you can add age/sex if available from user
        })

    # Call interpreter (import fallback handled in utils.cbc_interpreter.interpret_rows)
    try:
        interp_results = interpret_rows(rows, prefer_import=True)
        if interp_results is None or not isinstance(interp_results, list):
            raise RuntimeError("Interpreter returned invalid result; falling back to rule-based")
    except Exception as e:
        # Log error; fallback to error entries so zip() won't fail
        print("Interpreter call failed:", repr(e))
        interp_results = []
        for r in rows:
            interp_results.append({**r, "flag": "error", "text": f"Interpreter error: {e}"})

    # Persist results (transactional). Normalize text_en/text_ar like upload_pdf_report_view
    saved = []
    with transaction.atomic():
        for row_input, out in zip(rows, interp_results):
            # find CanonicalTest if test_code present
            ct = None
            if row_input.get('test_code'):
                try:
                    ct = CanonicalTest.objects.filter(code=row_input['test_code']).first()
                except Exception:
                    ct = None

            # prepare TestResult fields
            value_raw = row_input.get('value_raw')
            if value_raw is None and row_input.get('value') is not None:
                value_raw = str(row_input.get('value'))

            tr = TestResult.objects.create(
                document=None,
                user=request.user,
                test=ct,
                label_raw=row_input.get('test_name') or row_input.get('test_code') or "",
                value_raw=value_raw or "",
                value_numeric=row_input.get('value') if isinstance(row_input.get('value'), (int, float)) else None,
                unit=row_input.get('unit') or "",
                verified=True
            )

            # Normalize interpreter/model output into plain strings text_en/text_ar
            text_en = ""
            text_ar = ""
            raw_model = None

            if isinstance(out, dict):
                # prefer explicit keys
                text_en = out.get('text_en') or out.get('text') or out.get('explanation') or ""
                text_ar = out.get('text_ar') or out.get('explanation_ar') or ""
                raw_model = out.get('raw_model_output') or out.get('raw_output') or out.get('raw') or None
            else:
                # out may be string
                if isinstance(out, str):
                    text_en = out
                raw_model = None

            # If raw_model is dict, pull nested fields
            if isinstance(raw_model, dict):
                if not text_en:
                    text_en = raw_model.get('text_en') or raw_model.get('text') or raw_model.get('content') or ""
                if not text_ar:
                    text_ar = raw_model.get('text_ar') or raw_model.get('content_ar') or ""

            # If raw_model is string, attempt JSON parse or regex extraction
            if isinstance(raw_model, str) and (not text_en or not text_ar):
                try:
                    parsed = json.loads(raw_model)
                except Exception:
                    parsed = None
                if isinstance(parsed, dict):
                    text_en = text_en or parsed.get('text_en') or parsed.get('text') or ""
                    text_ar = text_ar or parsed.get('text_ar') or parsed.get('content_ar') or ""
                elif isinstance(parsed, list) and parsed:
                    first = parsed[0]
                    if isinstance(first, dict):
                        text_en = text_en or first.get('text_en') or first.get('text') or ""
                        text_ar = text_ar or first.get('text_ar') or first.get('content_ar') or ""
                else:
                    # crude regex fallback
                    try:
                        import re
                        m = re.search(r"text_en['\"]?\s*[:=]\s*['\"](.+?)['\"](?:,|\}|$)", raw_model)
                        if m:
                            text_en = text_en or m.group(1).strip()
                        m2 = re.search(r"text_ar['\"]?\s*[:=]\s*['\"](.+?)['\"](?:,|\}|$)", raw_model)
                        if m2:
                            text_ar = text_ar or m2.group(1).strip()
                    except Exception:
                        pass

            # last fallbacks
            text_en = (text_en or "").strip()
            text_ar = (text_ar or "").strip()

            # Prepare meta: preserve original out dict if present
            meta_obj = {}
            if isinstance(out, dict):
                # shallow copy to avoid mutating original
                try:
                    meta_obj = out.copy()
                except Exception:
                    meta_obj = {"raw": str(out)}
            else:
                meta_obj = {"raw": str(out)}

            if text_ar:
                meta_obj['text_ar'] = text_ar

            ti = TestInterpretation.objects.create(
                test_result=tr,
                document=None,
                created_by=request.user,
                flag=(out.get('flag') if isinstance(out, dict) else 'unknown') or 'unknown',
                text=text_en,
                meta=meta_obj
            )

            saved.append({
                "test_result": tr,
                "interpretation": ti,
                "input": row_input,
                "output": out,
                "text_en": text_en,
                "text_ar": meta_obj.get('text_ar', "")
            })

    # Also prepare backward-compatible results list if needed by older template
    results = []
    for rec in saved:
        tr = rec['test_result']
        it = rec['interpretation']
        inp = rec['input']
        results.append({
            "test_name": tr.test.display_name if tr.test else tr.label_raw,
            "label_raw": tr.label_raw,
            "value_raw": tr.value_raw,
            "unit": tr.unit,
            "ref_min": inp.get('ref_min'),
            "ref_max": inp.get('ref_max'),
            "flag": it.flag,
            "text": it.text
        })

    # Render template: provide both new 'saved' structure and old 'results' for compatibility
    return render(request, 'manual_entry_result.html', {"saved": saved, "results": results})
# CBC_App/views.py  (add imports at top of file)
import os
import json
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_http_methods
from django.contrib import messages
from django.db import transaction

from .forms import PdfUploadForm
from .models import UploadDocument, TestResult, TestInterpretation, CanonicalTest
from .utils.pdf_processor import get_data_from_user, deepseek_extract_rows, fallback_extract_rows_by_canonical
from .utils.cbc_interpreter import interpret_rows

import json
import re
import tempfile
from typing import Any, Dict, Optional

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.db import transaction
from django.shortcuts import render
from django.views.decorators.http import require_http_methods

from .forms import PdfUploadForm
from .models import (
    UploadDocument, CanonicalTest,
    TestResult, TestInterpretation,
    # optional ProcessingLog if you have it
)




def upload_pdf_report_view(request):
    """
    Upload PDF -> extract text -> extract rows -> call interpreter -> persist results (with text_en/text_ar)
    """
    if request.method == "POST":
        form = PdfUploadForm(request.POST, request.FILES)
        if form.is_valid():
            f = form.cleaned_data['file']

            # create UploadDocument record first
            doc = UploadDocument.objects.create(
                user=request.user,
                file=f,
                filename=getattr(f, "name", "uploaded.pdf"),
                status="pending"
            )
            # ensure file saved to storage path so doc.file.path works
            doc.save()

            # read text from PDF (try to use local path if possible)
            try:
                file_path = doc.file.path
            except Exception:
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                for chunk in f.chunks():
                    tmp.write(chunk)
                tmp.flush()
                tmp.close()
                file_path = tmp.name

            # extract raw text (this uses your get_data_from_user implementation)
            try:
                raw_text = get_data_from_user(file_path)
            except Exception as e:
                doc.status = 'failed'
                doc.save()
                messages.error(request, f"Failed to extract text from PDF: {e}")
                return render(request, 'upload_pdf.html', {'form': form})

            doc.raw_text = (raw_text or "")[:200000]  # store truncated raw text
            doc.status = "processing"
            doc.save()

            # prepare canonical test list for fallback extractor
            canonical_qs = CanonicalTest.objects.all().order_by('display_name')
            canonical_names = list(canonical_qs.values_list('display_name', flat=True))

            # attempt deepseek extraction first (deepseek_extract_rows should return list of row dicts or None)
            try:
                rows = deepseek_extract_rows(raw_text)
            except Exception:
                rows = None

            if rows is None:
                rows = fallback_extract_rows_by_canonical(raw_text, canonical_names)

            if not rows:
                doc.status = 'failed'
                doc.save()
                messages.error(request, "Could not extract any test rows from the PDF.")
                return render(request, 'upload_pdf.html', {'form': form})

            # Interpreter call (wrapper that may use adapter/import/subprocess)
            try:
                interp_results = interpret_rows(rows, prefer_import=True)
                if not isinstance(interp_results, list):
                    raise RuntimeError("interpret_rows returned invalid output")
            except Exception as e:
                interp_results = []
                for r in rows:
                    interp_results.append({**r, "flag": "error", "text": f"Interpreter error: {e}"})

            # Persist results (transactional). Normalize English/Arabic fields from interpreter output.
            saved = []
            with transaction.atomic():
                doc.status = 'completed'
                doc.save()

                for row_input, out in zip(rows, interp_results):
                    # map to CanonicalTest where possible (case-insensitive exact name)
                    ct = CanonicalTest.objects.filter(display_name__iexact=(row_input.get('test_name') or "")).first()

                    # create TestResult
                    value_raw = row_input.get('value_raw')
                    if value_raw is None and row_input.get('value') is not None:
                        value_raw = str(row_input.get('value'))
                    tr = TestResult.objects.create(
                        document=doc,
                        user=request.user,
                        test=ct,
                        value_raw=value_raw or "",
                        value_numeric=row_input.get('value') if isinstance(row_input.get('value'), (int, float)) else None,
                        unit=row_input.get('unit') or "",
                        verified=True
                    )

                    # Normalize interpreter/model output into plain strings text_en/text_ar
                    text_en = ""
                    text_ar = ""

                    # out might be dict-like or string. Prefer explicit keys.
                    if isinstance(out, dict):
                        # direct keys
                        text_en = out.get('text_en') or out.get('text') or out.get('explanation') or ""
                        text_ar = out.get('text_ar') or out.get('explanation_ar') or ""
                        raw_model = out.get('raw_model_output') or out.get('raw_output') or out.get('raw') or None
                    else:
                        raw_model = None
                        # if out is string, use it as english fallback
                        if isinstance(out, str):
                            text_en = out

                    # If raw_model is a dict, try to pull inner fields
                    if isinstance(raw_model, dict):
                        if not text_en:
                            text_en = raw_model.get('text_en') or raw_model.get('text') or raw_model.get('content') or ""
                        if not text_ar:
                            text_ar = raw_model.get('text_ar') or raw_model.get('content_ar') or ""

                    # If raw_model is a string, try json parse or crude regex
                    if isinstance(raw_model, str) and (not text_en or not text_ar):
                        try:
                            parsed = json.loads(raw_model)
                        except Exception:
                            parsed = None
                        if isinstance(parsed, dict):
                            text_en = text_en or parsed.get('text_en') or parsed.get('text') or ""
                            text_ar = text_ar or parsed.get('text_ar') or parsed.get('content_ar') or ""
                        elif isinstance(parsed, list) and parsed:
                            first = parsed[0]
                            if isinstance(first, dict):
                                text_en = text_en or first.get('text_en') or first.get('text') or ""
                                text_ar = text_ar or first.get('text_ar') or first.get('content_ar') or ""
                        else:
                            # crude regex fallback
                            try:
                                import re
                                m = re.search(r"text_en['\"]?\s*[:=]\s*['\"](.+?)['\"](?:,|\}|$)", raw_model)
                                if m:
                                    text_en = text_en or m.group(1).strip()
                                m2 = re.search(r"text_ar['\"]?\s*[:=]\s*['\"](.+?)['\"](?:,|\}|$)", raw_model)
                                if m2:
                                    text_ar = text_ar or m2.group(1).strip()
                            except Exception:
                                pass

                    # final coercion -> ensure plain strings
                    text_en = (text_en or "")
                    text_ar = (text_ar or "")

                    # save TestInterpretation (store english in text field; keep arabic inside meta)
                    meta_obj = {}
                    if isinstance(out, dict):
                        meta_obj = out.copy()
                    # attach text_ar inside meta for future retrieval
                    if text_ar:
                        meta_obj['text_ar'] = text_ar
                    else:
                        # try to add from raw_model if present
                        if isinstance(raw_model, dict) and raw_model.get('text_ar'):
                            meta_obj['text_ar'] = raw_model.get('text_ar')

                    ti = TestInterpretation.objects.create(
                        test_result=tr,
                        document=doc,
                        created_by=request.user,
                        flag=(out.get('flag') if isinstance(out, dict) else 'unknown') or 'unknown',
                        text=text_en,
                        meta=meta_obj
                    )

                    saved.append({
                        "test_result": tr,
                        "interpretation": ti,
                        "input": row_input,
                        "output": out,
                        "text_en": text_en,
                        "text_ar": meta_obj.get('text_ar', "")
                    })

            # render results template (saved now contains text_en/text_ar strings)
            return render(request, 'pdf_extract_result.html', {'doc': doc, 'saved': saved, 'form': form})

    else:
        form = PdfUploadForm()

    return render(request, 'upload_pdf.html', {'form': form})

from django.contrib.auth.decorators import login_required
from django.views.decorators.http import require_POST, require_http_methods
from django.shortcuts import render, get_object_or_404, redirect
from django.http import JsonResponse, HttpResponseForbidden
from django.db import transaction
from django.utils import timezone

# import models/forms
from .models import SharedReport, DoctorRecommendation, UserProfile, TestResult, UploadDocument
from .forms import ShareWithDoctorForm, DoctorRecommendationForm
from django.contrib.auth import get_user_model
User = get_user_model()

@login_required
def doctors_list_api(request):
    """
    Return JSON list of doctors (id, full_name, specialty if available).
    Can be used by modal to populate doctors dropdown.
    """
    qs = UserProfile.objects.filter(role='doctor').select_related('user').all()
    doctors = []
    for p in qs:
        doctors.append({
            "id": p.user.id,
            "username": p.user.username,
            "full_name": p.full_name or p.user.get_full_name() or p.user.username,
            # add specialty if you have a field, else empty
            "specialty": getattr(p, 'specialty', '') if hasattr(p, 'specialty') else ''
        })
    return JsonResponse({"doctors": doctors})

from django.http import JsonResponse, HttpResponseForbidden
from django.shortcuts import get_object_or_404
from django.db import transaction
import uuid

@require_POST
@login_required
def share_with_doctor(request):
    """
    AJAX endpoint to create a SharedReport.
    POST body: doctor_id, test_ids (comma separated), document_id (optional), message.
    Accepts test_ids as comma-separated UUID strings or numeric ids.
    """
    form = ShareWithDoctorForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"ok": False, "errors": form.errors}, status=400)

    # Only patients can share (enforce policy)
    try:
        profile = request.user.userprofile
    except Exception:
        profile = None
    if profile and profile.role != 'patient':
        return HttpResponseForbidden("Only patients can share reports with doctors.")

    # doctor id (assume numeric user PK)
    doctor_id = form.cleaned_data['doctor_id']
    test_ids_raw = form.cleaned_data.get('test_ids') or ""
    document_id = form.cleaned_data.get('document_id') or None
    message = form.cleaned_data.get('message') or ''

    # validate doctor exists and is a doctor
    doctor = get_object_or_404(User, id=doctor_id)
    try:
        if not doctor.userprofile.is_doctor():
            return JsonResponse({"ok": False, "error": "Selected user is not a doctor."}, status=400)
    except Exception:
        return JsonResponse({"ok": False, "error": "Selected user profile invalid."}, status=400)

    # parse test ids: accept UUIDs or ints; keep original strings for filtering
    raw_parts = [p.strip() for p in (test_ids_raw or "").split(',') if p.strip()]
    if raw_parts:
        # try to coerce to UUID if possible, else keep as string/int
        parsed_ids = []
        for p in raw_parts:
            # if it's numeric, keep as int (for projects using integer PKs on TestResult)
            if p.isdigit():
                try:
                    parsed_ids.append(int(p))
                    continue
                except Exception:
                    pass
            # try UUID
            try:
                parsed_ids.append(uuid.UUID(p))  # will produce UUID object
                continue
            except Exception:
                # keep string fallback (some DB backends accept uuid string)
                parsed_ids.append(p)

    else:
        parsed_ids = []

    # Create SharedReport and attach tests that belong to the current user
    with transaction.atomic():
        shared = SharedReport.objects.create(
            sender=request.user,
            doctor=doctor,
            message=message,
            document = UploadDocument.objects.filter(id=document_id).first() if document_id else None
        )

        added_count = 0
        if parsed_ids:
            # Build a filter: id__in parsed_ids. Django handles UUIDs and strings/ints as appropriate.
            tests_qs = TestResult.objects.filter(id__in=parsed_ids, user=request.user)
            # Add them to M2M
            for t in tests_qs:
                shared.tests.add(t)
                added_count += 1

        shared.save()

    # Return helpful response
    resp = {"ok": True, "shared_id": str(shared.id), "tests_added": added_count}
    if parsed_ids and added_count == 0:
        resp['warning'] = "No matching test rows found for the provided IDs (they must belong to you)."

    return JsonResponse(resp)



@login_required
def doctor_inbox_view(request):
    """
    Page for doctors to see shared reports sent to them.
    """
    # ensure doctor
    try:
        if not request.user.userprofile.is_doctor():
            return HttpResponseForbidden("Only doctors can access this page.")
    except Exception:
        return HttpResponseForbidden("Profile missing.")

    q = SharedReport.objects.filter(doctor=request.user).select_related('sender', 'document').prefetch_related('tests', 'recommendations')
    return render(request, 'doctor_inbox.html', {"shared_reports": q})


import json
from django.utils import timezone

@login_required
def shared_detail_view(request, shared_id):
    """
    Show shared report details (doctor and patient can view).
    Prepares safe 'tests_info' where each item contains:
      - test_result (TestResult instance)
      - flag (string)
      - text_en (string)
      - text_ar (string)
    This avoids fragile lookups inside templates.
    """
    shared = get_object_or_404(
        SharedReport.objects.select_related('sender','doctor','document')
            .prefetch_related('tests__interpretations', 'recommendations'),
        id=shared_id
    )

    # permission: either sender (patient) or target doctor or admin
    if not (request.user == shared.doctor or request.user == shared.sender or (hasattr(request.user,'is_staff') and request.user.is_staff)):
        return HttpResponseForbidden("Not authorized")

    # mark viewed if doctor opened it
    if request.user == shared.doctor and shared.status == 'pending':
        shared.status = 'viewed'
        shared.viewed_at = timezone.now()
        shared.save()

    # Build tests_info list (safe values for template)
    tests_info = []
    for t in shared.tests.all():
        # get most-recent interpretation (or first)
        interp = t.interpretations.order_by('created_at').first()
        flag = None
        text_en = ""
        text_ar = ""

        if interp:
            # direct text column (we store English in TestInterpretation.text)
            if interp.text:
                # prefer stored text (assumed english)
                text_en = str(interp.text).strip()
            # meta field may be dict or string
            meta = interp.meta or {}
            if isinstance(meta, str):
                try:
                    meta = json.loads(meta)
                except Exception:
                    # leave meta as string fallback
                    meta = {"raw": str(meta)}
            # helper to safely extract nested keys
            def _get_from_meta(keys):
                cur = meta
                for k in keys:
                    if isinstance(cur, dict) and k in cur:
                        cur = cur[k]
                    else:
                        return None
                return cur if cur is not None else None

            # try various known places for english/ar text
            if not text_en:
                text_en = meta.get('text_en') or meta.get('text') or _get_from_meta(['raw_model_output','text_en']) or _get_from_meta(['raw_model_object','text_en']) or ""
            text_ar = meta.get('text_ar') or _get_from_meta(['raw_model_output','text_ar']) or _get_from_meta(['raw_model_object','text_ar']) or ""

            # flag fallback
            if not flag:
                flag = interp.flag or meta.get('flag') or _get_from_meta(['raw_model_output','flag']) or _get_from_meta(['raw_model_object','flag'])

        # if no interp, but maybe some inline metadata on TestResult? (rare)
        if not flag:
            # try to read flag from t (if stored)
            try:
                # maybe previous adapters set last flag in t.meta (not in model by default)
                # keep None if not found
                pass
            except Exception:
                pass

        tests_info.append({
            "test_result": t,
            "flag": flag or "",
            "text_en": (text_en or "").strip(),
            "text_ar": (text_ar or "").strip(),
        })

    rec_form = DoctorRecommendationForm()
    return render(request, 'shared_detail.html', {
        "shared": shared,
        "rec_form": rec_form,
        "tests_info": tests_info,
    })
 

@require_POST
@login_required
def add_recommendation(request):
    form = DoctorRecommendationForm(request.POST)
    if not form.is_valid():
        return JsonResponse({"ok": False, "errors": form.errors}, status=400)

    shared_id = form.cleaned_data['shared_report_id']
    text = form.cleaned_data['text']
    visible = form.cleaned_data.get('visible_to_patient', True)

    shared = get_object_or_404(SharedReport, id=shared_id)

    # only assigned doctor can add recommendation
    if request.user != shared.doctor:
        return HttpResponseForbidden("Only assigned doctor can add recommendations.")

    with transaction.atomic():
        rec = DoctorRecommendation.objects.create(
            shared_report=shared,
            doctor=request.user,
            text=text,
            visible_to_patient=visible
        )
        shared.status = 'responded'
        shared.save()

    return JsonResponse({"ok": True, "recommendation_id": str(rec.id), "text": rec.text})
    

@login_required
def my_shared_reports_view(request):
    """
    Patient's view: list of reports they have shared and doctor's replies.
    """
    q = SharedReport.objects.filter(sender=request.user).select_related('doctor','document').prefetch_related('recommendations','tests')
    return render(request, 'my_shared_list.html', {"shared_reports": q})



@require_POST
@login_required
def shared_accept(request):
    shared_id = request.POST.get('shared_id')
    shared = get_object_or_404(SharedReport, id=shared_id)
    if request.user != shared.doctor:
        return JsonResponse({'ok': False, 'error':'forbidden'}, status=403)
    shared.status = 'viewed'
    shared.viewed_at = timezone.now()
    shared.save()
    return JsonResponse({'ok': True})
@require_POST
@login_required
def shared_reject(request):
    shared_id = request.POST.get('shared_id')
    shared = get_object_or_404(SharedReport, id=shared_id)
    if request.user != shared.doctor:
        return JsonResponse({'ok': False, 'error':'forbidden'}, status=403)
    shared.status = 'archived'
    shared.viewed_at = timezone.now()
    shared.save()
    return JsonResponse({'ok': True})
