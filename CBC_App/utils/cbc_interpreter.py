
# # CBC_Main_Project\CBC_App\utils\cbc_interpreter.py
# import os
# import sys
# import json
# import subprocess
# from pathlib import Path
# from typing import List, Dict, Any

# BASE_DIR = Path(__file__).resolve().parent.parent  # CBC_App/
# INTERPRETER_APP_DIR = os.path.abspath(os.path.join(BASE_DIR, "interpreter_src", "app"))
# INTERPRET_CLI_PATH = os.path.join(INTERPRETER_APP_DIR, "interpret_cli.py")


# def _safe_decode(b: bytes) -> str:
#     try:
#         return b.decode('utf-8')
#     except UnicodeDecodeError:
#         try:
#             return b.decode('latin-1')
#         except Exception:
#             return b.decode('utf-8', errors='replace')


# def _extract_first_json_array(s: str) -> str:
#     """
#     Find the first balanced JSON array substring in s and return it.
#     Raises ValueError if not found.
#     """
#     start = s.find('[')
#     if start == -1:
#         raise ValueError("No '[' found in stdout")
#     depth = 0
#     in_str = False
#     escape = False
#     for i in range(start, len(s)):
#         ch = s[i]
#         if ch == '"' and not escape:
#             in_str = not in_str
#         if ch == '\\' and in_str:
#             escape = not escape
#             continue
#         else:
#             escape = False
#         if not in_str:
#             if ch == '[':
#                 depth += 1
#             elif ch == ']':
#                 depth -= 1
#                 if depth == 0:
#                     return s[start:i+1]
#     raise ValueError("Could not find matching closing ']' for JSON array")


# def interpret_rows_subprocess(rows: List[Dict[str, Any]], cli_path: str = INTERPRET_CLI_PATH, timeout: int = 30) -> List[Dict[str, Any]]:
#     if not os.path.exists(cli_path):
#         raise FileNotFoundError(f"interpret_cli.py not found at {cli_path}")

#     env = os.environ.copy()
#     env['PYTHONIOENCODING'] = 'utf-8'
#     env.setdefault('LANG', 'en_US.UTF-8')
#     env.setdefault('LC_ALL', 'en_US.UTF-8')

#     proc = subprocess.Popen(
#         [sys.executable, cli_path],
#         stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
#         env=env
#     )
#     payload = json.dumps(rows, ensure_ascii=False).encode('utf-8')
#     try:
#         out, err = proc.communicate(payload, timeout=timeout)
#     except subprocess.TimeoutExpired:
#         proc.kill()
#         out, err = proc.communicate()
#         raise RuntimeError("Interpreter CLI timed out")

#     out_text = _safe_decode(out)
#     err_text = _safe_decode(err)
#     if proc.returncode != 0 and not out_text:
#         # if CLI exited with error and no stdout, raise
#         raise RuntimeError(f"Interpreter CLI exited with code {proc.returncode}. stderr: {err_text}")

#     # try direct parse
#     try:
#         parsed = json.loads(out_text)
#         if not isinstance(parsed, list):
#             raise ValueError("Parsed JSON is not a list")
#         return parsed
#     except Exception:
#         # attempt to extract first JSON array substring
#         try:
#             arr = _extract_first_json_array(out_text)
#             parsed = json.loads(arr)
#             if not isinstance(parsed, list):
#                 raise ValueError("Extracted JSON is not a list")
#             return parsed
#         except Exception as e:
#             # fallback: return per-row error with diagnostic snippet
#             fallback = []
#             for r in rows:
#                 fallback.append({
#                     **(r or {}),
#                     "flag": "error",
#                     "text": f"Interpreter output parsing failed: {e}. Raw stdout snippet: {out_text[:400]}"
#                 })
#             return fallback


# def interpret_rows_import(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """
#     Try to import adapter from interpreter_src/app and call interpret_rows(rows).
#     Must return a list of dicts or raise.
#     """
#     if INTERPRETER_APP_DIR not in sys.path:
#         sys.path.insert(0, INTERPRETER_APP_DIR)
#     try:
#         import adapter
#     except Exception as e:
#         raise ImportError(f"Failed to import adapter from {INTERPRETER_APP_DIR}: {e}")
#     if not hasattr(adapter, 'interpret_rows'):
#         raise AttributeError("adapter.interpret_rows not found")
#     out = adapter.interpret_rows(rows)
#     if out is None or not isinstance(out, list):
#         raise RuntimeError("adapter.interpret_rows returned invalid result")
#     return out


# def interpret_rows(rows: List[Dict[str, Any]], prefer_import: bool = True) -> List[Dict[str, Any]]:
#     """
#     Unified entrypoint called by Django.
#     Tries import method first (fast), falls back to subprocess CLI.
#     Always returns a list of dicts (one per input row).
#     """
#     if prefer_import:
#         try:
#             return interpret_rows_import(rows)
#         except Exception:
#             # fallback to subprocess quietly
#             return interpret_rows_subprocess(rows)
#     else:
#         return interpret_rows_subprocess(rows)


# cbc_interpreter.py (thin wrapper to new interpreter_service)
from typing import List, Dict, Any

# we keep the same function name used in views.py
try:
    from .interpreter_service import interpret_rows as _interpret_rows_impl
except Exception as e:
    # if import fails, provide a fallback rule-based stub
    _interpret_rows_impl = None

def interpret_rows(rows: List[Dict[str, Any]], prefer_import: bool = True) -> List[Dict[str, Any]]:
    """
    Wrapper kept for backward compatibility with views.py.
    Delegates to interpreter_service.interpret_rows when available.
    """
    if _interpret_rows_impl is None:
        # simple inline fallback (very small rule-based)
        out = []
        for r in rows:
            name = r.get('test_name') or r.get('test_code') or "Test"
            unit = r.get('unit') or ""
            try:
                val = float(r.get('value'))
                rmin = r.get('ref_min')
                rmax = r.get('ref_max')
                if rmin is not None and rmax is not None:
                    if val < float(rmin):
                        flag = "low"
                        text = f"{name} is low ({val} {unit})."
                    elif val > float(rmax):
                        flag = "high"
                        text = f"{name} is high ({val} {unit})."
                    else:
                        flag = "normal"
                        text = f"{name} is within reference range ({rmin}–{rmax} {unit})."
                else:
                    flag = "unknown"
                    text = f"{name}: {val} {unit}"
            except Exception:
                flag = "unknown"
                text = f"{name}: value not numeric"
            merged = dict(r)
            merged.update({"flag": flag, "text": text})
            out.append(merged)
        return out
    # delegate to new implementation
    return _interpret_rows_impl(rows, prefer_import=prefer_import)
