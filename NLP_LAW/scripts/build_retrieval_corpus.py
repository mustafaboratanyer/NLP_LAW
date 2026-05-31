import argparse
import json
import re
import unicodedata
from pathlib import Path

import fitz


LAW_METADATA = {
    "1.3.2004.pdf": {
        "law_name": "İcra ve İflas Kanunu",
        "law_no": "2004",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.3.2004.pdf",
    },
    "1.3.6183.pdf": {
        "law_name": "Amme Alacaklarının Tahsil Usulü Hakkında Kanun",
        "law_no": "6183",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.3.6183.pdf",
    },
    "1.3.7201.pdf": {
        "law_name": "Tebligat Kanunu",
        "law_no": "7201",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.3.7201.pdf",
    },
    "1.4.193.pdf": {
        "law_name": "Gelir Vergisi Kanunu",
        "law_no": "193",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.4.193.pdf",
    },
    "1.4.213.pdf": {
        "law_name": "Vergi Usul Kanunu",
        "law_no": "213",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.4.213.pdf",
    },
    "1.5.1136.pdf": {
        "law_name": "Avukatlık Kanunu",
        "law_no": "1136",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.1136.pdf",
    },
    "1.5.1512.pdf": {
        "law_name": "Noterlik Kanunu",
        "law_no": "1512",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.1512.pdf",
    },
    "1.5.2577.pdf": {
        "law_name": "İdari Yargılama Usulü Kanunu",
        "law_no": "2577",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.2577.pdf",
    },
    "1.5.2709.pdf": {
        "law_name": "Türkiye Cumhuriyeti Anayasası",
        "law_no": "2709",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.2709.pdf",
    },
    "1.5.2942.pdf": {
        "law_name": "Kamulaştırma Kanunu",
        "law_no": "2942",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.2942.pdf",
    },
    "1.5.3065.pdf": {
        "law_name": "Katma Değer Vergisi Kanunu",
        "law_no": "3065",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.3065.pdf",
    },
    "1.5.3071.pdf": {
        "law_name": "Dilekçe Hakkının Kullanılmasına Dair Kanun",
        "law_no": "3071",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.3071.pdf",
    },
    "1.5.3194.pdf": {
        "law_name": "İmar Kanunu",
        "law_no": "3194",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.3194.pdf",
    },
    "1.5.4721.pdf": {
        "law_name": "Türk Medeni Kanunu",
        "law_no": "4721",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.4721.pdf",
    },
    "1.5.4734.pdf": {
        "law_name": "Kamu İhale Kanunu",
        "law_no": "4734",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.4734.pdf",
    },
    "1.5.4735.pdf": {
        "law_name": "Kamu İhale Sözleşmeleri Kanunu",
        "law_no": "4735",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.4735.pdf",
    },
    "1.5.4857.pdf": {
        "law_name": "İş Kanunu",
        "law_no": "4857",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.4857.pdf",
    },
    "1.5.4982.pdf": {
        "law_name": "Bilgi Edinme Hakkı Kanunu",
        "law_no": "4982",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.4982.pdf",
    },
    "1.5.5237.pdf": {
        "law_name": "Türk Ceza Kanunu",
        "law_no": "5237",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.5237.pdf",
    },
    "1.5.5271.pdf": {
        "law_name": "Ceza Muhakemesi Kanunu",
        "law_no": "5271",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.5271.pdf",
    },
    "1.5.5275.pdf": {
        "law_name": "Ceza ve Güvenlik Tedbirlerinin İnfazı Hakkında Kanun",
        "law_no": "5275",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.5275.pdf",
    },
    "1.5.5326.pdf": {
        "law_name": "Kabahatler Kanunu",
        "law_no": "5326",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.5326.pdf",
    },
    "1.5.5393.pdf": {
        "law_name": "Belediye Kanunu",
        "law_no": "5393",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.5393.pdf",
    },
    "1.5.6098.pdf": {
        "law_name": "Türk Borçlar Kanunu",
        "law_no": "6098",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.6098.pdf",
    },
    "1.5.6284.pdf": {
        "law_name": "Ailenin Korunması ve Kadına Karşı Şiddetin Önlenmesine Dair Kanun",
        "law_no": "6284",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.6284.pdf",
    },
    "1.5.634.pdf": {
        "law_name": "Kat Mülkiyeti Kanunu",
        "law_no": "634",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.634.pdf",
    },
    "1.5.6100.pdf": {
        "law_name": "Hukuk Muhakemeleri Kanunu",
        "law_no": "6100",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.6100.pdf",
    },
    "1.5.6102.pdf": {
        "law_name": "Türk Ticaret Kanunu",
        "law_no": "6102",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.6102.pdf",
    },
    "1.5.6502.pdf": {
        "law_name": "Tüketicinin Korunması Hakkında Kanun",
        "law_no": "6502",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.6502.pdf",
    },
    "1.5.657.pdf": {
        "law_name": "Devlet Memurları Kanunu",
        "law_no": "657",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.657.pdf",
    },
    "1.5.6698.pdf": {
        "law_name": "Kişisel Verilerin Korunması Kanunu",
        "law_no": "6698",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.6698.pdf",
    },
    "1.5.5510.pdf": {
        "law_name": "Sosyal Sigortalar ve Genel Sağlık Sigortası Kanunu",
        "law_no": "5510",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.5510.pdf",
    },
    "1.5.6331.pdf": {
        "law_name": "İş Sağlığı ve Güvenliği Kanunu",
        "law_no": "6331",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.6331.pdf",
    },
    "1.5.6356.pdf": {
        "law_name": "Sendikalar ve Toplu İş Sözleşmesi Kanunu",
        "law_no": "6356",
        "source_url": "https://www.mevzuat.gov.tr/MevzuatMetin/1.5.6356.pdf",
    },
}


ARTICLE_HEADER_RE = re.compile(
    r"(?im)^\s*((?:(?:EK\s+GEÇİCİ|EK|GEÇİCİ|MÜKERRER)\s+)?MADDE\s+\d+[\/A-Za-zÇĞİÖŞÜçğıöşü]*|"
    r"(?:(?:Ek\s+Geçici|Ek|Geçici|Mükerrer)\s+)?Madde\s+\d+[\/A-Za-zÇĞİÖŞÜçğıöşü]*)\s*[-–—:]?"
)

APPENDIX_START_RE = re.compile(
    r"(?m)^\s*(?:\d{1,2}/\d{1,2}/\d{4}\s+TARİH(?:Lİ)?\s+VE\s+)?"
    r"(?:\d{3,5}\s+SAYILI\s+)?"
    r"(?:[A-ZÇĞİÖŞÜ0-9\s]+?\s+)?KANUN(?:A|DA|UN|UNA|UNDA)\s+"
    r"(?:EK\s+VE\s+DEĞİŞİKLİK|İŞLENEMEYEN|IŞLENEMEYEN|"
    r"\d+\s+NC[İI]\s+MADDESİNDEKİ\s+İDARİ\s+PARA\s+CEZALARI\s+"
    r"MİKTARLARIYLA\s+İLGİLİ\s+TABLO)"
    ,
    re.IGNORECASE,
)
CETVEL_START_RE = re.compile(r"(?m)^\s*(?:[IVXLCDM]+|\([IVXLCDM]+\))\s+SAYILI\s+CETVEL\b")
USULSUZLUK_CETVEL_START_RE = re.compile(
    r"(?im)^\s*\d+\s+Sayılı\s+Usulsüzlük\s+Cezalarına\s+Ait\s+Cetvel"
)
KROKI_APPENDIX_START_RE = re.compile(r"(?im)^\s*\(Ek\s+kroki\b")


def extract_pdf_text(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    pages = []
    for page in doc:
        pages.append(page.get_text("text"))
    return "\n".join(pages)


def clean_text(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?im)^(\s*(?:MADDE|Madde)\s+\d+)l(?=\d)", r"\g<1>1", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_appendices(text: str) -> str:
    matches = [
        match
        for match in (
            APPENDIX_START_RE.search(text),
            CETVEL_START_RE.search(text),
            USULSUZLUK_CETVEL_START_RE.search(text),
            KROKI_APPENDIX_START_RE.search(text),
        )
        if match
    ]
    if matches:
        match = min(matches, key=lambda item: item.start())
        return text[: match.start()].strip()
    return text


def article_id_part(header: str) -> str:
    normalized = header.casefold().replace("ı", "i")
    normalized = unicodedata.normalize("NFKD", normalized)
    normalized = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    normalized = re.sub(r"[^a-z0-9/]+", "_", normalized).strip("_")
    return normalized


FOOTNOTE_START_RE = re.compile(
    r"^\s*\d{1,3}\s+(?:(?:[a-zçğıöşü]\s*-\s*)?\d{1,2}/\d{1,2}/\d{4}\b|"
    r"Bu\s+|"
    r"Bu\s+Kanun(?:da|un)|"
    r"Bu\s+üst\s+başlık|"
    r"Anayasa\s+Mahkemesi|Resmi\s+Gazete|Söz\s+konusu)",
    re.IGNORECASE,
)


def is_heading_line(line: str) -> bool:
    line = line.strip()
    line_for_detection = re.sub(r"\d{1,6}$", "", line).strip()
    if line_for_detection:
        line = line_for_detection

    if not line or len(line) >= 100:
        return False
    if re.search(r"\d{1,2}/\d{1,2}/\d{4}", line):
        return False

    words = line.split()
    letters = [ch for ch in line if ch.isalpha()]
    has_lowercase = any(ch.islower() for ch in letters)

    if re.match(r"^[IVXLCDM]+\.\s+", line):
        return True
    if re.match(r"^[IVXLCDM]+\s*[–—-]\s+", line):
        return True
    if re.match(r"^[A-ZÇĞİÖŞÜ]\.\s+", line):
        return True
    if re.match(r"^[a-zçğıöşü]\.\s+", line):
        return True
    if re.match(r"^[A-ZÇĞİÖŞÜ]\)\s+", line):
        return len(words) <= 12 and not line.endswith((".", ";", ","))
    if re.match(r"^[a-zçğıöşü]\)\s+", line):
        return line.endswith(":") and len(words) <= 15
    if re.match(r"^\d+\s*[–—-]\s+", line):
        return is_heading_line(re.sub(r"^\d+\s*[–—-]\s+", "", line, count=1))
    if line.endswith(":"):
        if re.match(r"^\d+\s+", line):
            return len(words) <= 14
        return 1 <= len(words) <= 12 and line[:1].isupper()
    if line.endswith((".", ";", ",", ")")):
        return False
    if letters and not has_lowercase:
        return True
    if re.match(r"^\d+\.\s+[A-Za-zÇĞİÖŞÜçğıöşü]", line):
        return True
    if 1 <= len(words) <= 8 and line[:1].isupper() and not re.search(r"\d", line):
        return True

    return False


def looks_like_article_title(title: str) -> bool:
    if is_heading_line(title):
        return True

    words = title.split()
    if not 1 <= len(words) <= 22:
        return False
    if not title[:1].isupper():
        return False
    if title.endswith((".", ";", ",", ")")):
        return False
    if re.search(r"\d{1,2}/\d{1,2}/\d{4}", title):
        return False
    if re.search(
        r"\b(?:uygulanır|edilir|olunur|belirlenir|yetkilidir|yapılır|kalkar|"
        r"verilir|alınır|bulunur|sayılır|istenir)\b$",
        title.casefold(),
    ):
        return False

    return True


def looks_like_high_level_heading(line: str) -> bool:
    line = line.strip()
    if not line:
        return True
    if re.search(r"\d{1,2}/\d{1,2}/\d{4}", line):
        return True
    if re.match(r"^[A-ZÇĞİÖŞÜ\s]+$", line):
        return True
    if re.search(r"\b(?:KISIM|BÖLÜM|AYIRIM|KİTAP)\b", line, flags=re.IGNORECASE):
        return True
    return False


def looks_like_section_title(line: str) -> bool:
    section_markers = (
        "hükümler",
        "hükümleri",
        "görev, yetki ve yükümlülükleri",
        "kapsamdaki kişiler ve tescili",
        "prim alınması, prime esas",
        "sosyal sigorta hükümleri",
        "konsey, kurul ve koordinasyon",
        "teftiş ve idari yaptırımlar",
        "kuruluş esasları ve organlar",
        "kuruluşların gelirleri, denetimi ve kapatılması",
        "toplu iş sözleşmesinin genel esasları",
        "toplu iş sözleşmesinin yapılması",
        "yüksek hakem kurulunun kuruluşu ve çalışma esasları",
    )
    normalized = line.casefold()
    folded = normalized.replace("ı", "i")
    folded = unicodedata.normalize("NFKD", folded)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded_markers = []
    for marker in section_markers:
        folded_marker = marker.casefold().replace("ı", "i")
        folded_marker = unicodedata.normalize("NFKD", folded_marker)
        folded_markers.append("".join(ch for ch in folded_marker if not unicodedata.combining(ch)))
    return any(marker in normalized or marker in folded for marker in section_markers + tuple(folded_markers))


INLINE_TRAILING_TITLE_RE = re.compile(r"\s+[-–—]\s*([^–—\n]{2,160}:)\s*$")
UNNUMBERED_TRAILING_ARTICLE_RE = re.compile(
    r"(?ims)^\s*(?:Ek\s+Geçici\s+Madde|EK\s+GEÇİCİ\s+MADDE)\s*[-–—].*\Z"
)


def normalize_article_title(title: str) -> str:
    title = remove_inline_footnote_refs(title)
    title = " ".join(title.split()).strip()
    title = re.sub(r"(?<=[A-Za-zÇĞİÖŞÜçğıöşü])\d{1,12}(?=\s|$)", "", title)
    title = re.sub(r"\d{1,6}$", "", title).strip()
    if not title or len(title) > 260:
        return ""
    if title.casefold() in {"md.", "md.)", "md)"}:
        return ""
    if title.endswith((".", ";", ",")):
        return ""
    if title.endswith(")") and re.search(r"(?:tarihli|kararı|e:|k:)", title, re.IGNORECASE):
        return ""
    if title.endswith("(...)"):
        return ""
    if re.search(r"\d+/\d+.*\bmd\.?\)?$", title, re.IGNORECASE):
        return ""
    if re.search(r"(?:ibaresi|maddesiyle|madde metninden|metne işlendiği|işlenmiştir)", title, re.IGNORECASE):
        return ""
    if re.match(r"^(?:MADDE|Madde|GEÇİCİ|Geçici|EK|Ek|MÜKERRER|Mükerrer)\b", title):
        return ""
    return title


def embedded_title_from_line(line: str) -> str:
    match = INLINE_TRAILING_TITLE_RE.search(line.strip())
    if not match:
        return ""

    title = normalize_article_title(match.group(1))
    if title and is_heading_line(title):
        return title
    return ""


def strip_embedded_trailing_title(line: str) -> str:
    match = INLINE_TRAILING_TITLE_RE.search(line)
    if not match:
        return line

    title = normalize_article_title(match.group(1))
    if not title or not is_heading_line(title):
        return line

    prefix = line[: match.start()].rstrip()
    if re.fullmatch(r"(?:\d+|[IVXLCDM]+|[A-Za-zÇĞİÖŞÜçğıöşü])\.?", prefix.strip()):
        return ""
    return prefix


def extract_article_title(text: str, header_start: int, header: str) -> str:
    prefix = remove_inline_footnote_refs(remove_footnote_blocks(text[:header_start])).rstrip()
    if not prefix:
        return ""

    lines = prefix.splitlines()
    candidates = []

    for line in reversed(lines):
        stripped = line.strip()
        if not stripped:
            if candidates:
                break
            continue

        embedded_title = embedded_title_from_line(stripped)
        if embedded_title and not ARTICLE_HEADER_RE.match(stripped):
            return embedded_title

        candidates.append(stripped)
        if len(candidates) >= 12:
            break

    candidates = list(reversed(candidates))
    header_indexes = [
        idx
        for idx, line in enumerate(candidates)
        if ARTICLE_HEADER_RE.match(line.strip())
    ]
    if header_indexes:
        candidates = candidates[header_indexes[-1] + 1 :]
        if not candidates:
            return ""

    if len(candidates) > 1:
        section_indexes = [idx for idx, line in enumerate(candidates[:-1]) if looks_like_section_title(line)]
        if section_indexes:
            candidates = candidates[section_indexes[-1] + 1 :] or candidates

    if len(candidates) > 1:
        sentence_indexes = [
            idx
            for idx, line in enumerate(candidates[:-1])
            if re.search(r"[.;)]$", line.strip()) and not is_heading_line(line)
        ]
        if sentence_indexes:
            tail_candidates = candidates[sentence_indexes[-1] + 1 :]
            if any(looks_like_article_title(normalize_article_title(line)) for line in tail_candidates):
                candidates = tail_candidates

    filtered_candidates = [line for line in candidates if not looks_like_high_level_heading(line)]
    if filtered_candidates:
        candidates = filtered_candidates

    if len(candidates) > 1:
        last_title = normalize_article_title(candidates[-1])
        if last_title.endswith(":") and looks_like_article_title(last_title):
            return last_title

    raw_combined = " ".join(candidates)
    if raw_combined.endswith(":") and len(raw_combined.split()) <= 45:
        return normalize_article_title(raw_combined)

    combined = " ".join(candidates)

    if combined.endswith(":") and len(combined.split()) <= 45:
        return normalize_article_title(combined)

    while len(candidates) > 1 and len(candidates[0]) < 25 and not candidates[0].endswith((".", ":", ";")):
        candidates.pop(0)

    title = normalize_article_title(" ".join(candidates))
    if title and looks_like_article_title(title):
        return title
    return ""


def trailing_block_start(lines: list[str]) -> int | None:
    for index in range(len(lines) - 1, -1, -1):
        if not lines[index].strip():
            return index + 1
    return None


def is_trailing_title_block(block: list[str]) -> bool:
    clean_block = [line.strip() for line in block if line.strip()]
    if not clean_block or len(clean_block) > 12:
        return False
    if sum(len(line) for line in clean_block) > 300:
        return False
    combined = " ".join(clean_block)
    if combined.endswith(":") and len(combined.split()) <= 45:
        return True
    if any(re.search(r"\d{1,2}/\d{1,2}/\d{4}", line) for line in clean_block):
        return False
    if any(re.search(r"[.;,)]$", line) for line in clean_block):
        return False
    if not any(is_heading_line(line) for line in clean_block):
        return False
    return True


def strip_trailing_noise(text: str) -> str:
    lines = text.rstrip().splitlines()

    while lines:
        changed = False

        while lines and not lines[-1].strip():
            lines.pop()
            changed = True

        if lines:
            stripped_line = strip_embedded_trailing_title(lines[-1])
            if stripped_line != lines[-1]:
                if stripped_line.strip():
                    lines[-1] = stripped_line
                else:
                    lines.pop()
                changed = True
                while lines and not lines[-1].strip():
                    lines.pop()

        while len(lines) > 1 and is_heading_line(lines[-1]):
            lines.pop()
            changed = True
            while lines and not lines[-1].strip():
                lines.pop()

        block_start = trailing_block_start(lines)
        if block_start and is_trailing_title_block(lines[block_start:]):
            lines = lines[:block_start]
            changed = True

        if changed:
            continue

        break

    text = "\n".join(lines).strip()
    return UNNUMBERED_TRAILING_ARTICLE_RE.sub("", text).strip()


def strip_known_trailing_title(text: str, title: str) -> str:
    title = normalize_article_title(title)
    if not title:
        return text.strip()

    title_norm = " ".join(title.split())
    lines = text.rstrip().splitlines()

    while lines and not lines[-1].strip():
        lines.pop()

    block_start = trailing_block_start(lines)
    if block_start and block_start < len(lines):
        block_lines = [line.strip() for line in lines[block_start:] if line.strip()]
        block_norm = " ".join(" ".join(block_lines).split())
        block_title_norm = normalize_article_title(block_norm) or block_norm
        if block_title_norm == title_norm or block_title_norm.endswith(title_norm):
            return "\n".join(lines[:block_start]).strip()

    return "\n".join(lines).strip()


def remove_footnote_blocks(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []
    index = 0

    while index < len(lines):
        if FOOTNOTE_START_RE.match(lines[index]):
            index += 1
            while index < len(lines) and lines[index].strip():
                index += 1
            continue

        cleaned_lines.append(lines[index])
        index += 1

    return "\n".join(cleaned_lines).strip()


def remove_inline_footnote_refs(text: str) -> str:
    text = re.sub(
        r"([.;:,])\s*\d{1,12}"
        r"(?!\s*(?:ve|sayılı|sayili|tarihli|inci|ıncı|uncu|üncü|nci|ncı|ncu|ncü)\b)"
        r"(?=\s|$)",
        r"\1",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"(\))\d{1,12}(?=\s|$)", r"\1", text)
    return text


def split_articles(text: str) -> list[dict[str, str]]:
    matches = list(ARTICLE_HEADER_RE.finditer(text))
    articles = []

    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        header = match.group(1).strip()
        title = extract_article_title(text, start, header)
        next_title = ""
        if index + 1 < len(matches):
            next_match = matches[index + 1]
            next_title = extract_article_title(text, next_match.start(), next_match.group(1).strip())
        article_text = text[start:end].strip()
        article_text = remove_footnote_blocks(article_text)
        article_text = remove_inline_footnote_refs(article_text)
        article_text = deduplicate_leading_header(header, article_text)
        article_text = strip_known_trailing_title(article_text, next_title)
        article_text = strip_trailing_noise(article_text)

        if title:
            article_text = f"{title}\n{article_text}"

        if len(article_text) < 40:
            continue

        articles.append(
            {
                "article_no": header,
                "article_key": article_id_part(header),
                "article_title": title,
                "text": article_text,
            }
        )

    return articles


def deduplicate_leading_header(header: str, text: str) -> str:
    escaped = re.escape(header)
    duplicate_header = re.compile(rf"^\s*{escaped}\s+(?={escaped}\s*[-–—:]?)", re.IGNORECASE)
    text = duplicate_header.sub("", text)
    stacked_header = re.compile(rf"^\s*{escaped}\s*\n+\s*{escaped}\s*[-–—:]?\s*", re.IGNORECASE)
    return stacked_header.sub(f"{header} - ", text).strip()


def build_corpus(raw_dir: Path, law_no: str | None = None) -> list[dict[str, str]]:
    corpus = []
    seen_ids = set()

    for pdf_path in sorted(raw_dir.glob("*.pdf")):
        metadata = LAW_METADATA.get(pdf_path.name)
        if metadata is None:
            if law_no is None:
                print(f"Skipping unknown PDF: {pdf_path.name}")
            continue
        if law_no is not None and metadata["law_no"] != law_no:
            continue

        raw_text = extract_pdf_text(pdf_path)
        cleaned = strip_appendices(clean_text(raw_text))
        articles = split_articles(cleaned)

        for article in articles:
            base_corpus_id = f"{metadata['law_no']}_{article['article_key']}"
            corpus_id = base_corpus_id
            suffix = 2
            while corpus_id in seen_ids:
                corpus_id = f"{base_corpus_id}_{suffix}"
                suffix += 1

            seen_ids.add(corpus_id)
            corpus.append(
                {
                    "id": corpus_id,
                    "law_name": metadata["law_name"],
                    "law_no": metadata["law_no"],
                    "article_no": article["article_no"],
                    "article_title": article["article_title"],
                    "text": article["text"],
                    "source_url": metadata["source_url"],
                }
            )

        print(f"{pdf_path.name}: {len(articles)} articles")

    return corpus


def main() -> None:
    parser = argparse.ArgumentParser(description="Build article-level legal retrieval corpus from PDFs.")
    parser.add_argument(
        "--raw-dir",
        default=r"C:\Users\Alver\Documents\data\raw",
        help="Directory containing Mevzuat PDF files.",
    )
    parser.add_argument(
        "--out",
        default="data/processed/retrieval_corpus.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--law-no",
        default=None,
        help="Optional law number to process only one PDF, e.g. 6100.",
    )
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    corpus = build_corpus(raw_dir, law_no=args.law_no)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(corpus)} corpus chunks to {out_path}")


if __name__ == "__main__":
    main()
