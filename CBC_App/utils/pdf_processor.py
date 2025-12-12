# CBC_App/utils/pdf_processor.py
import os
import re
import json
from typing import List, Dict, Any, Optional
from PyPDF2 import PdfReader

# optional: deepseek via hf router
try:
    from openai import OpenAI
    _HAS_OPENAI = True
except Exception:
    OpenAI = None
    _HAS_OPENAI = False


def get_data_from_user(data_input: str) -> str:
    """
    Open the PDF file path and extract text from the first page.
    Returns extracted text string.
    """
    text = ""
    with open(data_input, 'rb') as f:
        reader = PdfReader(f)
        # some PDFs have 0 pages guard
        if len(reader.pages) >= 1:
            page = reader.pages[0]
            text = page.extract_text() or ""
    return text


def FIND_USABLE_DATA(final_get_data: str, query: str) -> str:
    """
    Original FIND_USABLE_DATA logic you provided.
    It finds query in large text and returns a processed substring where
    multiple spaces are replaced by "----" marker (old behavior).
    """
    if not final_get_data or not query:
        return ""

    pos = final_get_data.find(query)
    if pos == -1:
        return ""

    REQUESTED_INFORMATION = ""
    INIT = 0
    while (pos + INIT) < len(final_get_data):
        ch = final_get_data[pos + INIT]
        if ch == "\n":
            break
        REQUESTED_INFORMATION += ch
        INIT += 1

    NEW_DATA = ""
    counter_MELON = 0
    for i in REQUESTED_INFORMATION:
        if counter_MELON == 1:
            NEW_DATA = NEW_DATA + "----"
            counter_MELON = 2

        if i != " ":
            NEW_DATA = NEW_DATA + i
            counter_MELON = 0
            continue

        counter_MELON += 1

    return NEW_DATA


# ---------- Helper: extract numbers/units from a small text chunk ----------
_VALUE_RE = re.compile(r"(?P<value>[-+]?\d{1,3}(?:[,\d]*\d)?(?:[.,]\d+)?)(?:\s*(?P<unit>%|g/dL|mg/dL|fL|pg|/mm3|mill/cumm|cumm|mil/mm3|mm3)?)?", re.IGNORECASE)

def parse_value_unit_from_chunk(chunk: str) -> Dict[str, Optional[str]]:
    """
    Tough heuristic: find first numeric token and optional unit.
    Returns dict {'value_raw','value','unit'}
    """
    res = {"value_raw": None, "value": None, "unit": None}
    if not chunk:
        return res
    m = _VALUE_RE.search(chunk)
    if m:
        raw = m.group("value")
        unit = m.group("unit") or ""
        # normalize decimal comma
        raw_norm = raw.replace(',', '')
        try:
            value = float(raw_norm)
        except Exception:
            try:
                value = float(raw_norm.replace(',', '.'))
            except Exception:
                value = None
        res["value_raw"] = raw
        res["value"] = value
        res["unit"] = unit.strip()
    return res


# ---------- DeepSeek / HF router extractor ----------
def deepseek_extract_rows(raw_text: str, model_name: str = "deepseek-ai/DeepSeek-V3.2-Exp:novita") -> Optional[List[Dict[str,Any]]]:
    """
    Ask the DeepSeek model to parse the raw OCR text into JSON array of rows:
    Each row: { test_name, value_raw, value (number), unit, ref_min, ref_max, meta }
    Returns list or None on failure.
    """
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token or not _HAS_OPENAI:
        return None

    prompt = (
        "You are an assistant that extracts lab test rows from noisy OCR text. "
        "Given the raw text below, return a JSON array only (no extra text). "
        "Each array element must be an object with keys: test_name, value_raw, value, unit, ref_min, ref_max, meta.\n\n"
        "If a field is not available, set it to null. Example element:\n"
        '{"test_name":"Hemoglobin","value_raw":"12.5","value":12.5,"unit":"g/dL","ref_min":13.0,"ref_max":17.0,"meta":{}}\n\n'
        "Now parse the following text:\n\n"
        f"RAW_TEXT:\n{raw_text}\n\nReturn only JSON array."
    )
    try:
        client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user","content":prompt}],
            # optional: temperature=0.0 to maximize determinism
        )
        # robustly read text from response (matching different SDK shapes)
        text = None
        try:
            choice0 = resp.choices[0]
            msg = getattr(choice0, "message", None)
            if msg:
                text = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
            if text is None:
                # dict-like fallback
                text = choice0.get("message", {}).get("content") if isinstance(choice0, dict) else None
            if text is None:
                text = str(resp)
        except Exception:
            try:
                text = resp["choices"][0]["message"]["content"]
            except Exception:
                text = str(resp)
        # extract first JSON array in text (robust)
        arr_start = text.find('[')
        if arr_start == -1:
            return None
        # try incremental parsing until array closes
        depth = 0
        in_str = False
        escape = False
        end_index = None
        for i in range(arr_start, len(text)):
            ch = text[i]
            if ch == '"' and not escape:
                in_str = not in_str
            if ch == '\\' and in_str:
                escape = not escape
                continue
            else:
                escape = False
            if not in_str:
                if ch == '[':
                    depth += 1
                elif ch == ']':
                    depth -= 1
                    if depth == 0:
                        end_index = i
                        break
        if end_index is None:
            return None
        arr_text = text[arr_start:end_index+1]
        parsed = json.loads(arr_text)
        if not isinstance(parsed, list):
            return None
        # normalize elements: ensure keys exist
        out = []
        for p in parsed:
            if not isinstance(p, dict):
                continue
            out.append({
                "test_name": p.get("test_name") or p.get("test") or p.get("label") or None,
                "value_raw": str(p.get("value_raw")) if p.get("value_raw") is not None else (str(p.get("value")) if p.get("value") is not None else None),
                "value": (float(p.get("value")) if p.get("value") is not None else None),
                "unit": p.get("unit") or None,
                "ref_min": (float(p.get("ref_min")) if p.get("ref_min") is not None else None),
                "ref_max": (float(p.get("ref_max")) if p.get("ref_max") is not None else None),
                "meta": p.get("meta") or {}
            })
        return out
    except Exception as e:
        # model call failed
        return None


# ---------- Fallback simple extractor based on canonical tests ----------
def fallback_extract_rows_by_canonical(raw_text: str, canonical_tests: List[str]) -> List[Dict[str,Any]]:
    """
    For each canonical test name, call FIND_USABLE_DATA to get a chunk and parse the first numeric value.
    canonical_tests is a list of display_name strings from DB.
    Returns rows list with keys like in deepseek output.
    """
    rows = []
    lower_text = raw_text or ""
    for t in canonical_tests:
        chunk = FIND_USABLE_DATA(lower_text, t)
        if not chunk:
            # try case-insensitive search
            chunk = FIND_USABLE_DATA(lower_text, t.lower())
        parsed = parse_value_unit_from_chunk(chunk)
        rows.append({
            "test_name": t,
            "value_raw": parsed.get("value_raw"),
            "value": parsed.get("value"),
            "unit": parsed.get("unit"),
            "ref_min": None,
            "ref_max": None,
            "meta": {"extracted_chunk": chunk}
        })
    # Optionally filter out rows with no value (keep only where value present)
    rows = [r for r in rows if r.get("value") is not None]
    return rows

# # CBC_App/utils/pdf_processor.py
# import os
# import re
# import json
# from typing import List, Dict, Any, Optional
# from PyPDF2 import PdfReader

# # محاولة استدعاء عميل OpenAI (Router => HuggingFace)
# try:
#     from openai import OpenAI
#     _HAS_OPENAI = True
# except Exception:
#     OpenAI = None
#     _HAS_OPENAI = False

# # --- قراءة كل صفحات PDF ---
# def extract_text_pages(file_path: str) -> List[str]:
#     """
#     Return a list of page texts (page 0, page1, ...).
#     """
#     pages = []
#     with open(file_path, "rb") as f:
#         reader = PdfReader(f)
#         for p in reader.pages:
#             text = p.extract_text() or ""
#             pages.append(text)
#     return pages


# # --- heuristic: استنتاج نوع التقرير من نص الصفحة ---
# REPORT_KEYWORDS = {
#     "CBC": ["complete blood count", "complete blood picture", "cbc", "hematology", "blood complete picture"],
#     "Biochemistry": ["glucose", "lipid", "cholesterol", "creatinine", "uric acid", "biochemistry"],
#     "Urine": ["urine", "urinalysis"],
#     "Coagulation": ["pt", "inr", "aptt", "coagulation"],
#     # أضف مفاتيح أخرى حسب حاجةك
# }

# def infer_report_type_from_text(text: str) -> str:
#     if not text:
#         return "Unknown"
#     lower = text.lower()
#     for label, keys in REPORT_KEYWORDS.items():
#         for k in keys:
#             if k in lower:
#                 return label
#     # fallback: detect lab name or header line
#     m = re.search(r'(^[A-Z][A-Za-z\s]{2,40})\n', text)
#     if m:
#         return m.group(1).strip()
#     return "Unknown"


# # --- DeepSeek: استخراج مُجمّع متعدد-الصفحات (يتوقع أن كل عنصر يحتوي report_type) ---
# def deepseek_extract_rows_multi(full_text: str, model_name: str = "deepseek-ai/DeepSeek-V3.2-Exp:novitata") -> Optional[List[Dict[str,Any]]]:
#     """
#     Ask remote model to return JSON array of rows with keys:
#     test_name, value_raw, value, unit, ref_min, ref_max, meta, report_type
#     Returns list or None on failure.
#     """
#     hf_token = os.environ.get('HF_TOKEN')
#     if not hf_token or not _HAS_OPENAI:
#         return None

#     prompt = (
#         "You are a parser. Given noisy OCR text of a multi-page lab PDF, return only a JSON array.\n"
#         "Each element must be an object with keys: test_name, value_raw, value (number|null), unit, ref_min, ref_max, meta (object), report_type (string).\n"
#         "Group tests that belong to the same report section under same report_type. If you cannot determine reference, use null.\n\n"
#         "Example item:\n"
#         '{"test_name":"Hemoglobin","value_raw":"16.6","value":16.6,"unit":"g/dL","ref_min":13.0,"ref_max":17.0,"meta":{},"report_type":"CBC"}\n\n'
#         "Now parse the text below and return only the JSON array.\n\n"
#         f"RAW_TEXT:\n{full_text}\n\nReturn only JSON array."
#     )
#     try:
#         client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)
#         resp = client.chat.completions.create(
#             model=model_name,
#             messages=[{"role":"user","content":prompt}],
#             temperature=0.0
#         )
#         # robust extraction of response text
#         text = None
#         try:
#             choice0 = resp.choices[0]
#             msg = getattr(choice0, "message", None)
#             if msg:
#                 text = getattr(msg, "content", None) or (msg.get("content") if isinstance(msg, dict) else None)
#             if text is None:
#                 text = choice0.get("message", {}).get("content") if isinstance(choice0, dict) else None
#             if text is None:
#                 text = str(resp)
#         except Exception:
#             try:
#                 text = resp["choices"][0]["message"]["content"]
#             except Exception:
#                 text = str(resp)

#         # extract first JSON array substring (balanced)
#         start = text.find('[')
#         if start == -1:
#             return None
#         depth = 0
#         in_str = False
#         escape = False
#         end_index = None
#         for i in range(start, len(text)):
#             ch = text[i]
#             if ch == '"' and not escape:
#                 in_str = not in_str
#             if ch == '\\' and in_str:
#                 escape = not escape
#                 continue
#             else:
#                 escape = False
#             if not in_str:
#                 if ch == '[':
#                     depth += 1
#                 elif ch == ']':
#                     depth -= 1
#                     if depth == 0:
#                         end_index = i
#                         break
#         if end_index is None:
#             return None
#         arr_text = text[start:end_index+1]
#         parsed = json.loads(arr_text)
#         out = []
#         for p in parsed:
#             # normalization: ensure report_type key
#             out.append({
#                 "test_name": p.get("test_name") or p.get("label") or None,
#                 "value_raw": p.get("value_raw") if p.get("value_raw") is not None else (str(p.get("value")) if p.get("value") is not None else None),
#                 "value": float(p.get("value")) if p.get("value") is not None else None,
#                 "unit": p.get("unit") or None,
#                 "ref_min": float(p.get("ref_min")) if p.get("ref_min") is not None else None,
#                 "ref_max": float(p.get("ref_max")) if p.get("ref_max") is not None else None,
#                 "meta": p.get("meta") or {},
#                 "report_type": p.get("report_type") or "Unknown"
#             })
#         return out
#     except Exception:
#         return None


# # --- fallback: استخراج صفحة-بصفحة ثم تجميع ---
# _VALUE_RE = re.compile(r"(?P<value>[-+]?\d{1,3}(?:[,\d]*\d)?(?:[.,]\d+)?)(?:\s*(?P<unit>%|g/dL|mg/dL|fL|pg|/mm3|mil/mm3|cumm|mill/cumm)?)?", re.IGNORECASE)

# def parse_value_unit_from_chunk(chunk: str) -> Dict[str,Optional[Any]]:
#     if not chunk:
#         return {"value_raw": None, "value": None, "unit": None}
#     m = _VALUE_RE.search(chunk)
#     if not m:
#         return {"value_raw": None, "value": None, "unit": None}
#     raw = m.group("value")
#     unit = m.group("unit") or None
#     raw_norm = raw.replace(',', '')
#     try:
#         value = float(raw_norm)
#     except Exception:
#         try:
#             value = float(raw_norm.replace(',', '.'))
#         except Exception:
#             value = None
#     return {"value_raw": raw, "value": value, "unit": (unit or "").strip() if unit else None}

# def fallback_extract_rows_multi(page_texts: List[str], canonical_names: List[str]) -> List[Dict[str,Any]]:
#     """
#     For each page: attempt to find canonical tests or numeric tokens; produce rows with report_type inferred per page.
#     Returns list of rows.
#     """
#     rows = []
#     for page_idx, text in enumerate(page_texts):
#         report_type = infer_report_type_from_text(text)
#         lower_text = text or ""
#         # try to find each canonical test in page
#         for cname in canonical_names:
#             # case-insensitive index
#             pos = lower_text.lower().find(cname.lower())
#             if pos != -1:
#                 # take a chunk after the found name (e.g., up to newline or 120 chars)
#                 start = pos
#                 endpos = min(len(lower_text), pos + 160)
#                 chunk = lower_text[start:endpos]
#                 parsed = parse_value_unit_from_chunk(chunk)
#                 if parsed.get("value") is not None:
#                     rows.append({
#                         "test_name": cname,
#                         "value_raw": parsed.get("value_raw"),
#                         "value": parsed.get("value"),
#                         "unit": parsed.get("unit"),
#                         "ref_min": None,
#                         "ref_max": None,
#                         "meta": {"page": page_idx, "extracted_chunk": chunk},
#                         "report_type": report_type
#                     })
#         # if no canonical found: try to find numeric lines (generic)
#         if not any(r["meta"].get("page") == page_idx for r in rows):
#             # find lines that contain numbers
#             for line in (text or "").splitlines():
#                 if re.search(r"\d", line):
#                     parsed = parse_value_unit_from_chunk(line)
#                     if parsed.get("value") is not None:
#                         # attempt to infer test_name from line (text before number)
#                         name_part = re.split(r"[-:\t\d]", line, 1)[0].strip()
#                         if len(name_part) > 0:
#                             rows.append({
#                                 "test_name": name_part,
#                                 "value_raw": parsed.get("value_raw"),
#                                 "value": parsed.get("value"),
#                                 "unit": parsed.get("unit"),
#                                 "ref_min": None,
#                                 "ref_max": None,
#                                 "meta": {"page": page_idx, "extracted_line": line},
#                                 "report_type": report_type
#                             })
#     return rows


# # --- تجميع بحسب report_type ---
# def group_rows_by_report_type(rows: List[Dict[str,Any]]) -> Dict[str, List[Dict[str,Any]]]:
#     grouped = {}
#     for r in rows:
#         typ = r.get("report_type") or "Unknown"
#         grouped.setdefault(typ, []).append(r)
#     return grouped
