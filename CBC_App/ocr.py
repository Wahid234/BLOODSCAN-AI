# CBC_App/ocr.py
import re
from PIL import Image, ImageOps, ImageFilter
import pytesseract
from pytesseract import Output
from rapidfuzz import process as rf_process, fuzz as rf_fuzz

# adjust these patterns if you have more/other names
TEST_PATTERNS = {
    "WBC": ["wbc", "white blood", "wbc count"],
    "RBC": ["rbc", "r b c", "rbc count"],
    "Hemoglobin": ["hemoglobin", "hgb", r"\bhb\b"],
    "Hematocrit": ["hematocrit", "hct"],
    "MCV": ["mcv"],
    "MCH": ["mch"],
    "MCHC": ["mchc"],
    "RDW-CV": ["rdw", "rdw-cv"],
    "Platelets": ["platelets", "platelet"],
    "Neutrophils": ["neutrophils", "neutrophil"],
    "Lymphocytes": ["lymphocytes","lymphocyte"],
    "Monocytes": ["monocytes","monocyte"],
    "Eosinophils": ["eosinophils","eosinophil"],
    "Basophils": ["basophils","basophil"],
    "Bands": ["bands"]
}

NUM_RE = re.compile(
    r"([0-9]{1,3}(?:[,\u00A0][0-9]{3})*(?:\.[0-9]+)?|\d+\.\d+|\d+)\s*(%|g/dL|g\/dL|fL|pg|/mm3|mil/mm3|/ul|/µL|/μL)?",
    re.IGNORECASE,
)
# CBC_App/ocr_helpers.py   (أو ضعها في ocr.py)
from rapidfuzz import process as rf_process, fuzz as rf_fuzz
from CBC_App.models import CanonicalTest

# load canonical names once (call at runtime inside view)
def load_canonical_choices():
    # return list of (display_name, id) or only names
    qs = CanonicalTest.objects.all().values_list('display_name', flat=True)
    return list(qs)

def fuzzy_map_label(label, canonical_names, threshold=90):
    """
    Map extracted label -> canonical display_name if score >= threshold.
    Returns (canonical_display_name, score) or (None, score).
    """
    if not label or not canonical_names:
        return (None, 0)
    # normalize
    q = label.strip().lower()
    match = rf_process.extractOne(q, canonical_names, scorer=rf_fuzz.token_sort_ratio)
    if not match:
        return (None, 0)
    matched_name, score, _ = match
    if score >= threshold:
        return (matched_name, score)
    return (None, score)

def preprocess_image(path, max_side=2200):
    """
    Basic preprocessing to improve OCR: grayscale, resize, autocontrast, sharpen, threshold.
    """
    im = Image.open(path).convert("L")
    w, h = im.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        im = im.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
    im = ImageOps.autocontrast(im)
    im = im.filter(ImageFilter.SHARPEN)
    # no hard threshold here; we'll let tesseract handle text segmentation. But a soft threshold can be used.
    return im

def run_ocr_tsv(image, psm=6, oem=3, lang='eng'):
    """
    Use pytesseract to get detailed TSV output (words + their bounding boxes).
    psm 6 = Assume a single uniform block of text.
    """
    custom_config = f'--oem {oem} --psm {psm}'
    data = pytesseract.image_to_data(image, lang=lang, config=custom_config, output_type=Output.DICT)
    # data is dict with keys: level, page_num, block_num, par_num, line_num, word_num, left, top, width, height, conf, text
    return data

def cluster_x_positions(x_centers, img_width):
    """
    Simple clustering of x centers into columns.
    strategy:
      - sort unique x_centers
      - compute gaps; break clusters where gap > threshold (proportional to image width)
    returns list of (cluster_min, cluster_max) column ranges sorted by left->right
    """
    if not x_centers:
        return []
    xs = sorted(set(x_centers))
    # threshold: a gap bigger than this indicates new column; adjust factor if needed
    gap_threshold = max(40, img_width * 0.10)  # 10% of width or 40px minimum
    clusters = []
    current = [xs[0]]
    for x in xs[1:]:
        if x - current[-1] > gap_threshold:
            clusters.append((min(current), max(current)))
            current = [x]
        else:
            current.append(x)
    clusters.append((min(current), max(current)))
    # expand each range slightly
    expanded = []
    for a, b in clusters:
        expanded.append((a - 20, b + 20))
    return expanded

def assign_word_to_column(xc, clusters):
    """
    Return cluster index for xc based on clusters ranges.
    """
    for i, (lo, hi) in enumerate(clusters):
        if lo <= xc <= hi:
            return i
    # if nothing matched, find nearest cluster by center distance
    centers = [ (lo+hi)/2 for (lo,hi) in clusters ]
    if centers:
        dists = [abs(xc-c) for c in centers]
        return int(dists.index(min(dists)))
    return 0

def parse_tsv_to_rows(data, img_width, img_height):
    """
    Convert pytesseract TSV data into structured rows with columns based on x clustering.
    Returns list of rows; each row is dict {col_index: 'joined words', ...} and contains y center.
    """
    n = len(data['text'])
    words = []
    x_centers = []
    # Collect words with useful confidence and text
    for i in range(n):
        text = str(data['text'][i]).strip()

        conf_val = data['conf'][i]
        if isinstance(conf_val, (int, float)):
            conf = int(conf_val)
        elif isinstance(conf_val, str) and conf_val.lstrip('-').isdigit():
            conf = int(conf_val)
        else:
            conf = -1

        if text == "" or conf < 20:
            continue

        left = int(data['left'][i])
        top = int(data['top'][i])
        width = int(data['width'][i])
        height = int(data['height'][i])
        xc = left + width/2
        yc = top + height/2
        words.append({'text': text, 'xc': xc, 'yc': yc, 'left': left, 'top': top, 'w': width, 'h': height})
        x_centers.append(xc)
    if not words:
        return []

    # cluster x positions into columns
    clusters = cluster_x_positions(x_centers, img_width)

    # group words by approximate line (y). compute line groups by sorting by yc and grouping gaps
    words_sorted = sorted(words, key=lambda w: (w['yc'], w['xc']))
    lines = []
    current_line = [words_sorted[0]]
    for w in words_sorted[1:]:
        if abs(w['yc'] - current_line[-1]['yc']) > max(10, w['h']*0.8, 12):
            # new line
            lines.append(current_line)
            current_line = [w]
        else:
            current_line.append(w)
    if current_line:
        lines.append(current_line)

    # For each line, assign words to columns
    rows = []
    for ln in lines:
        # sort words by x
        ln_sorted = sorted(ln, key=lambda w: w['xc'])
        row = {'y': sum(w['yc'] for w in ln_sorted)/len(ln_sorted), 'cols': {}}
        for w in ln_sorted:
            col_idx = assign_word_to_column(w['xc'], clusters)
            row['cols'].setdefault(col_idx, []).append(w['text'])
        # join words per column
        row['cols'] = {k: " ".join(v) for k, v in row['cols'].items()}
        rows.append(row)
    return rows

def fuzzy_map_label_to_canonical(label, canonical_names):
    """
    Map a free-label string to a canonical test name using rapidfuzz.
    canonical_names: list of display_name strings
    """
    q = label.lower().strip()
    if not q:
        return None
    match = rf_process.extractOne(q, canonical_names, scorer=rf_fuzz.token_sort_ratio)
    if match:
        name, score, _ = match
        if score >= 70:
            return name
    return None

def parse_ocr_text_advanced(image_path, lang='eng'):
    """
    Full pipeline:
     - preprocess image
     - run pytesseract with TSV output
     - build rows and columns using coordinates
     - extract test label from left column(s) and numeric from mid column
     - fuzzy-map labels to canonical test names (if CanonicalTest loaded externally)
    Returns dict { 'raw_text': raw_string, 'rows': rows_list, 'parsed': parsed_items }
    """
    im = preprocess_image(image_path)
    # raw text too
    try:
        raw_text = pytesseract.image_to_string(im, lang=lang, config='--oem 3 --psm 6')
    except Exception:
        raw_text = ""
    data = run_ocr_tsv(im, psm=6, oem=3, lang=lang)
    img_w, img_h = im.size
    rows = parse_tsv_to_rows(data, img_w, img_h)

    # load candidate canonical names if you want to auto-map (optional)
    canonical_names = []  # fill in from DB in view if needed

    parsed = []
    for row in rows:
        cols = row['cols']
        # Heuristic: left-most column index = min key
        if not cols:
            continue
        col_indices = sorted(cols.keys())
        # detect which column likely holds numeric results: take the column that contains numbers most often
        # quick heuristic: if there are >=2 cols, assume middle column (or right if 3)
        # We'll pick:
        if len(col_indices) == 1:
            left_col = col_indices[0]
            mid_col = col_indices[0]
            right_col = col_indices[0]
        elif len(col_indices) == 2:
            left_col = col_indices[0]
            mid_col = col_indices[1]
            right_col = col_indices[1]
        else:
            left_col = col_indices[0]
            mid_col = col_indices[1]  # middle
            right_col = col_indices[-1]

        label_text = cols.get(left_col, "").strip()
        value_text = cols.get(mid_col, "").strip()
        ref_text = cols.get(right_col, "").strip() if right_col != mid_col else ""

        # If mid column doesn't look numeric, try right column as value
        if not NUM_RE.search(value_text or "") and NUM_RE.search(ref_text or ""):
            value_text, ref_text = ref_text, value_text

        # cleanup value_text for commas/spaces etc
        num_match = NUM_RE.search(value_text or "")
        unit = ""
        val_raw = ""
        if num_match:
            val_raw = num_match.group(1).replace('\u00A0', '').replace(',', '')
            unit = (num_match.group(2) or "").strip()
        else:
            # if no number found, try find number anywhere in row (left/mid/right)
            combined = " ".join([cols.get(k,"") for k in col_indices])
            m2 = NUM_RE.search(combined)
            if m2:
                val_raw = m2.group(1).replace('\u00A0', '').replace(',', '')
                unit = (m2.group(2) or "").strip()

        parsed.append({
            'label': label_text,
            'value': val_raw,
            'unit': unit,
            'ref': ref_text,
            'row_y': row['y'],
            'cols': cols
        })

    return {'raw_text': raw_text, 'rows': rows, 'parsed': parsed}
