# interpreter_service.py
import os
import json
from typing import List, Dict, Any, Optional
from decimal import Decimal
import os
import json
import math
from typing import List, Dict, Any, Optional


# try to import OpenAI-compatible client

from openai import OpenAI
_HAS_OPENAI_CLIENT = True


# ---------- Configuration helper ----------
def _load_config_candidate() -> Dict[str, Any]:
    """
    Try reading config.yml from common locations (project root or current app),
    but prefer environment variable HF_TOKEN.
    """
    import yaml
    out = {}
    # prefer env
    hf_env = os.environ.get("HF_TOKEN")
    if hf_env:
        out['hf_token'] = hf_env
    # try config files (optional, not required)
    candidates = [
        os.path.join(os.getcwd(), "config.yml"),
        os.path.join(os.getcwd(), "CBC_App", "config.yml"),
    ]
    for p in candidates:
        try:
            if os.path.exists(p):
                with open(p, 'r', encoding='utf-8') as fh:
                    data = yaml.safe_load(fh) or {}
                    for k in ('hf_token', 'hf_model'):
                        if k in data and data[k]:
                            out.setdefault(k, data[k])
        except Exception:
            continue
    # also check interpreter_src config but as very low priority (you said avoid relying on it)
    try:
        p2 = os.path.join(os.getcwd(), "CBC_App", "interpreter_src", "config.yml")
        if os.path.exists(p2):
            import yaml as _y
            with open(p2, 'r', encoding='utf-8') as fh:
                data = _y.safe_load(fh) or {}
                for k in ('hf_token', 'hf_model'):
                    if k in data and data[k]:
                        out.setdefault(k, data[k])
    except Exception:
        pass
    return out

# rule-based fallback (kept simple)
def rule_interpret_row(r: Dict[str, Any]) -> Dict[str, Any]:
    name = r.get('test_name') or r.get('test_code') or "Test"
    unit = r.get('unit') or ""
    try:
        val = float(r.get('value'))
        rmin = r.get('ref_min')
        rmax = r.get('ref_max')
        if rmin is not None and rmax is not None:
            rminf = float(rmin); rmaxf = float(rmax)
            if val < rminf:
                return {"flag":"low", "text": f"{name} is low ({val} {unit}). Consider follow-up."}
            elif val > rmaxf:
                return {"flag":"high", "text": f"{name} is high ({val} {unit}). Consider follow-up."}
            else:
                return {"flag":"normal", "text": f"{name} is within reference range ({rminf}–{rmaxf} {unit})."}
        else:
            return {"flag":"unknown", "text": f"{name}: {val} {unit} (no reference range)."}
    except Exception:
        return {"flag":"unknown", "text": f"{name}: value not numeric or missing."}



# ---------------------------
# Helpers
# ---------------------------
def _extract_first_json_array(s: str) -> str:
    """Return first balanced JSON array substring found in s or raise ValueError."""
    start = s.find('[')
    if start == -1:
        raise ValueError("No '[' found in model output")
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
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
                    return s[start:i+1]
    raise ValueError("Could not find matching closing ']' for JSON array")

def _safe_get_message_text(resp) -> str:
    """
    Robustly extract text content from different SDK response shapes.
    """
    try:
        choice0 = resp.choices[0]
        # attribute-like (some SDKs)
        msg = getattr(choice0, "message", None)
        if msg:
            # try .content attribute or dict-like
            text = getattr(msg, "content", None)
            if text:
                return text
            if isinstance(msg, dict):
                return msg.get("content", str(resp))
        # try dict-like
        if isinstance(choice0, dict):
            return choice0.get("message", {}).get("content", str(resp))
    except Exception:
        pass
    # fallback str(resp)
    try:
        return str(resp)
    except Exception:
        return ""

# Build a compact prompt to request a JSON array for many rows
def _build_batch_prompt(rows: List[Dict[str, Any]]) -> str:
    """
    Compose a prompt that enumerates the rows and instructs the model to return
    a JSON array of interpretation objects, one per input row, in the same order.
    """
    lines = []
    lines.append("You are a concise medical assistant. For each input lab test, return in the output JSON array an object with these keys:")
    lines.append("  test_name, value_raw, value, unit, ref_min, ref_max, flag, text")
    lines.append("Flag must be one of: low, normal, high, unknown, error.")
    lines.append("If ref_min or ref_max are empty in input, search online for appropriate reference ranges.")
    lines.append(" - text_en: one short sentence, simple English, suitable for a patient. Say whether low/normal/high and a brief suggestion.")
    lines.append(" - text_ar: the Arabic translation of the same short sentence (Modern Standard Arabic).")
    lines.append("Return only the JSON array (no extra commentary, no code fences). The order must match the inputs.")

    # lines.append("Return **only** a JSON array (no additional commentary). The array MUST preserve the order of the input tests.")
    lines.append("")
    lines.append("Input (one test per line):")
    for i, r in enumerate(rows, start=1):
        tn = (r.get("test_name") or r.get("test_code") or "").strip()
        val = r.get("value_raw") if r.get("value_raw") is not None else (r.get("value") if r.get("value") is not None else "")
        unit = r.get("unit") or ""
        rmin = r.get("ref_min") if r.get("ref_min") is not None else ""
        rmax = r.get("ref_max") if r.get("ref_max") is not None else ""
        lines.append(f"{i}. test_name: {tn} || value_raw: {val} || unit: {unit} || ref_min: {rmin} || ref_max: {rmax}")
    lines.append("")
    lines.append("Example output element:")
    lines.append('{"test_name":"Hemoglobin","value_raw":"16.6","value":16.6,"unit":"g/dL","ref_min":13.0,"ref_max":17.0,"flag":"normal","text_en":"Hemoglobin is normal (16.6 g/dL).","text_ar":"الهيموغلوبين طبيعي (16.6 غ/دل)."}')
    return "\n".join(lines)

# ---------- DeepSeek (HuggingFace router) client ----------
def _build_prompt_for_row(r: Dict[str, Any]) -> str:
    name = r.get('test_name') or r.get('test_code') or "Test"
    unit = r.get('unit') or ""
    value = r.get('value')
    ref_min = r.get('ref_min')
    ref_max = r.get('ref_max')

    prompt = (
        f"You are a concise medical assistant. Explain the following laboratory test result "
        f"in one short sentence suitable for a patient (simple language). "
        f"Include whether it's low/normal/high and a short suggestion.\n\n"
        f"Test: {name}\n"
        f"Result: {value} {unit}\n"
    )
    if ref_min is not None or ref_max is not None:
        prompt += f"Reference range: {ref_min or '?'} - {ref_max or '?'} {unit}\n"
    prompt += "\nReturn only a brief single-sentence explanation."
    return prompt


# Batch call to DeepSeek (HF router) that returns list[dict]
def interpret_with_deepseek_batch(client: OpenAI, rows: List[Dict[str, Any]], model_name: str) -> List[Dict[str, Any]]:
    prompt = _build_batch_prompt(rows)
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000  # tune if needed
        )
        text = _safe_get_message_text(resp)
        # extract first JSON array substring
        arr_text = _extract_first_json_array(text)
        parsed = json.loads(arr_text)
        if not isinstance(parsed, list):
            raise ValueError("Model returned JSON but it is not a list")
        # normalize parsed items to expected keys and types
        prompt = _build_batch_prompt(rows)
    except Exception as e:
        raise RuntimeError(f" failed: {e}")
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role":"user", "content": prompt}],
            temperature=0.0,
            max_tokens=2000,  # tune according to expected length
        )
        text = _safe_get_message_text(resp)
        arr_text = _extract_first_json_array(text)
        parsed = json.loads(arr_text)
        if not isinstance(parsed, list):
            raise ValueError("Model returned JSON but it is not a list")

        out = []
        for obj in parsed:
            if not isinstance(obj, dict):
                out.append({"flag":"error", "text_en":"Invalid model item", "text_ar":"عنصر غير صالح من النموذج"})
                continue

            # prefer explicit bilingual fields; support older 'text' as fallback
            text_en = obj.get("text_en") or obj.get("text") or ""
            text_ar = obj.get("text_ar") or obj.get("text") or ""

            o = {
                "test_name": obj.get("test_name") or obj.get("test") or None,
                "value_raw": obj.get("value_raw") if obj.get("value_raw") is not None else obj.get("value"),
                "value": obj.get("value") if isinstance(obj.get("value"), (int,float)) else None,
                "unit": obj.get("unit") or None,
                "ref_min": obj.get("ref_min") if obj.get("ref_min") is not None else None,
                "ref_max": obj.get("ref_max") if obj.get("ref_max") is not None else None,
                "flag": obj.get("flag") or "unknown",
                "text_en": text_en,
                "text_ar": text_ar,
                "raw_model_object": obj
            }
            out.append(o)
        return out
    except Exception as e:
        raise RuntimeError(f"DeepSeek batch call failed: {e}")

# Main entrypoint used by views
def interpret_rows(rows: List[Dict[str, Any]], prefer_import: bool = True, batch_size: int = 25, model_name: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Interpret a list of rows in a single or few batched model calls.
    - rows: list of dicts with keys test_name, value, value_raw, unit, ref_min, ref_max, meta
    - batch_size: maximum rows per model call (tune for tokens / quota)
    Returns list of dicts (each input row merged with flag/text).
    """
    results: List[Dict[str, Any]] = []

    # Determine HF token / model
    hf_token = os.environ.get("HF_TOKEN") or None
    if model_name is None:
        model_name = os.environ.get("HF_MODEL") or "deepseek-ai/DeepSeek-V3.2-Exp:novita"

    use_deepseek = False
    client = None
    if hf_token and _HAS_OPENAI_CLIENT:
        try:
            client = OpenAI(base_url="https://router.huggingface.co/v1", api_key=hf_token)
            use_deepseek = True
        except Exception:
            use_deepseek = False

    # If we cannot use deepseek -> local rule-based per-row
    if not use_deepseek or client is None:
        for r in rows:
            out = rule_interpret_row(r)
            merged = dict(r); merged.update(out)
            results.append(merged)
        return results

    # Use batched calls
    total = len(rows)
    n_batches = math.ceil(total / batch_size)
    model_outputs: List[Dict[str, Any]] = []

    for i in range(n_batches):
        batch_rows = rows[i*batch_size:(i+1)*batch_size]
        try:
            batch_out = interpret_with_deepseek_batch(client, batch_rows, model_name=model_name)
            # Some models might return objects with slightly different shapes; ensure length matches
            if not isinstance(batch_out, list):
                raise RuntimeError("Batch call returned non-list")
            # If the model returned fewer entries, align by index and fill missing entries with error
            if len(batch_out) != len(batch_rows):
                # attempt to align by test_name where possible
                mapped = []
                for j, br in enumerate(batch_rows):
                    # find first parsed entry with same test_name (case-insensitive)
                    match = None
                    for candidate in batch_out:
                        if candidate.get("test_name") and br.get("test_name") and candidate.get("test_name").strip().lower() == br.get("test_name").strip().lower():
                            match = candidate; break
                    if match:
                        mapped.append(match)
                    else:
                        # fallback to same-index if exists else error
                        if j < len(batch_out):
                            mapped.append(batch_out[j])
                        else:
                            mapped.append({"flag":"error","text":"Model returned fewer items than expected."})
                batch_out = mapped
        except Exception as e:
            # If batch call fails, fall back to per-row rule-based (or optionally per-row model calls)
            batch_out = []
            for br in batch_rows:
                # try rule fallback but include diagnostic in text
                fallback = rule_interpret_row(br)
                fallback["text"] = f"{fallback['text']} (fallback: model error: {e})"
                batch_out.append(fallback)

        model_outputs.extend(batch_out)

    # Now merge model_outputs with input rows
    for r, out in zip(rows, model_outputs):
        merged = dict(r)
        # ensure out has flag/text
        flag = out.get("flag") or "unknown"
        text = out.get("text") or ""
        merged.update({"flag": flag, "text": text, "raw_model_output": out})
        results.append(merged)

    return results
