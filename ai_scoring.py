# ai_scoring.py
import json
import os
import re
import difflib
import requests
import time
import pandas as pd
import openai
import numpy as np
from sentence_transformers import SentenceTransformer, util
from pythainlp.tokenize import word_tokenize
from pythainlp.corpus import thai_words
from pythainlp.tag import pos_tag
from pythainlp.util import normalize
from sklearn.metrics.pairwise import cosine_similarity

# โหลดโมเดล
model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

# ✅ -----------------------------
# หา path ของไฟล์ json ในโฟลเดอร์ data
BASE_DIR = os.path.dirname(__file__)  # โฟลเดอร์ปัจจุบัน

# ---- thai_loanwords ----
json_path = os.path.join(BASE_DIR, "data", "thai_loanwords_new_update.json")
try:
    with open(json_path, "r", encoding="utf-8") as f:
        thai_loanwords = json.load(f)
    loanwords_whitelist = {
        item["thai_word"] for item in thai_loanwords if "thai_word" in item
    }
except FileNotFoundError:
    print(f"⚠️ ไม่พบไฟล์: {json_path}")
    thai_loanwords = []
    loanwords_whitelist = set()

# ---- common misspellings ----
misspellings_path = os.path.join(BASE_DIR, "data", "update_common_misspellings.json")

try:
    with open(misspellings_path, "r", encoding="utf-8") as f:
        _cm = json.load(f)
    COMMON_MISSPELLINGS = {item["wrong"]: item["right"] for item in _cm}
except FileNotFoundError:
    print(f"⚠️ ไม่พบไฟล์: {misspellings_path} (COMMON_MISSPELLINGS ตั้งเป็น empty dict)")
    COMMON_MISSPELLINGS = {}

API_KEY = '33586c7cf5bfa0029887a9831bf94963' # add Apikey
API_URL = 'https://api.longdo.com/spell-checker/proof'

custom_words = {"ประเทศไทย", "สถาบันการศึกษา", "นานาประการ"}

#คำที่สามารถฉีกคำได้
splitable_phrases = {
    'แม้ว่า', 'ถ้าแม้ว่า', 'แต่ถ้า', 'แต่ทว่า', 'เนื่องจาก', 'ดังนั้น', 'เพราะฉะนั้น','ตกเป็น','เป็นการ',
    'ดีแต่', 'หรือไม่', 'ข้อมูลข่าวสาร', 'ทั่วโลก', 'ยังมี', 'ทำให้เกิด', 'เป็นโทษ', 'ไม่มี', 'ข้อควรระวัง', 'การแสดงความคิดเห็น', 'ผิดกฎหมาย', 'แสดงความคิดเห็น'
}
#คำที่ไม่สามารถฉีกคำได้
strict_not_split_words = {
    'มากมาย', 'ประเทศไทย', 'ออนไลน์', 'ความคิดเห็น', 'ความน่าเชื่อถือ'
}

thai_dict = set(w for w in set(thai_words()).union(custom_words) if (' ' not in w) and w.strip())

# allowed punctuation (เพิ่ม ' และ ")
allowed_punctuations = {'.', ',', '-', '(', ')', '!', '?', '%', '“', '”', '‘', '’', '"', "'", '…', 'ฯ'}

# Allow / Forbid list ไม้ยมก (เพิ่มคำที่ใช้บ่อย)
allow_list = {'ปี', 'อื่น', 'เล็ก', 'ใหญ่', 'มาก', 'หลาย', 'ช้า', 'เร็ว', 'ชัด', 'ดี', 'ผิด'}
forbid_list = {'นา', 'บางคน', 'บางอย่าง', 'บางสิ่ง', 'บางกรณี'}

explanations = [
    "1. ตรวจสอบการฉีกคำ",
    "2. ตรวจสอบคำสะกดผิดด้วย PyThaiNLP (และขอ Longdo ช่วยกรณีสงสัย)",
    "3. ตรวจสอบการใช้เครื่องหมายที่ไม่อนุญาต",
    "4. ตรวจสอบการใช้ไม้ยมก (ๆ) ถูกต้องตามบริบทหรือไม่",
    "5. ตรวจสอบการแยกคำผิด เช่น คำที่ควรติดกัน"
]
# ✅ -----------------------------

def normalize_text(text):
    text = " ".join(text.replace("\n", " ").replace("\r", " ").replace("\t", " ").split())
    return text.replace(" ", "")

def find_keywords_list(text, keywords):
    found = [kw for kw in keywords if kw.replace(" ", "") in text]
    return found

def score_group_1(text):
    text_norm = normalize_text(text)
    media_keywords = ["สื่อสังคมออนไลน์", "สื่อสังคม", "สื่อออนไลน์"]
    usage_keywords = ["เป็นช่องทาง", "ช่องทาง", "เป็นการแพร่กระจาย", "เป็นสื่อ", "สามารถ", "ทำให้", "เป็นการกระจาย", "นั้น"]
    last_keywords = ["แพร่กระจาย", "แพร่กระจายข่าวสาร", "ค้นหา", "รับข้อมูลข่าวสาร", "เผยแพร่", "ติดต่อสื่อสาร", "กระจาย", "รับสาร","รับรู้"]

    found_usage = [kw for kw in usage_keywords if kw.replace(" ", "") in text_norm]
    found_last = [kw for kw in last_keywords if kw.replace(" ", "") in text_norm]
    first_5_words = text.split()[:5]
    first_5_text = "".join(first_5_words)
    found_media_in_first_5 = any(kw in first_5_text for kw in media_keywords)

    score = 1 if (found_media_in_first_5 and found_usage and found_last) else 0
    return score

def score_group_2(text):
    text_norm = normalize_text(text)
    keypoints_1 = ["ไม่ระวัง", "ไม่ระมัดระวัง", "ขาดความรับผิดชอบ", "ควรระมัดระวัง", "ใช้ในทางที่ไม่ดี", "ไม่เหมาะสม", "อย่างระมัดระวัง", "ไตร่ตรอง"]
    keypoints_2 = [
        "โทษ", "ผลเสีย", "ข้อเสีย", "เกิดผลกระทบ", "สิ่งไม่ดี",
        "เสียหาย",
        "การเขียนแสดงความเห็นวิพากษ์วิจารณ์ผู้อื่นในทางเสียหาย",
        "การเขียนแสดงความคิดเห็นวิพากษ์วิจารณ์ผู้อื่นในทางเสียหาย",
        "ตกเป็นเหยื่อของมิจฉาชีพ",
        "ตกเป็นเหยื่อมิจฉาชีพ", "ตกเป็นเหยื่อทางการตลาด"
    ]
    found_1 = find_keywords_list(text_norm, keypoints_1)
    found_2 = find_keywords_list(text_norm, keypoints_2)
    found_illegal = "ผิดกฎหมาย" in text_norm

    score = 1 if (found_1 and found_2) or (found_1 and found_illegal and found_2) else 0
    return score

def score_group_3(text):
    text_norm = normalize_text(text)
    media_keypoint = ["สื่อสังคมออนไลน์", "สื่อสังคม", "สื่อออนไลน์"]
    keypoints = ["รู้เท่าทัน", "รู้ทัน", "ผู้ใช้ต้องรู้เท่าทัน", "รู้ทันสื่อสังคม",
                 "รู้เท่าทันสื่อ", "รู้ทันสื่อ", "สร้างภูมิคุ้มกัน", "ไม่ตกเป็นเหยื่อ", "แก้ปัญหาการตกเป็นเหยื่อ"]

    found_1 = find_keywords_list(text_norm, media_keypoint)
    found_2 = find_keywords_list(text_norm, keypoints)

    score = 1 if (found_1 and found_2) else 0
    return score

def score_group_4(text):
    text_norm = normalize_text(text)
    media_use_keywords = [
        "ใช้สื่อสังคม", "ใช้สื่อออนไลน์", "ใช้สื่อสังคมออนไลน์", "การใช้สื่อ"
    ]
    hidden_intent_keywords = ["เจตนาแอบแฝง"]
    effect_keywords = ["ผลกระทบต่อ", "ผลกระทบ"]
    credibility_keywords = [
        "ความน่าเชื่อถือของข่าวสาร", "ความน่าเชื่อถือของข้อมูลข่าวสาร", "ความน่าเชื่อถือของข้อมูล",
        "มีสติ", "ความน่าเชื่อถือ", "ความเชื่อถือของข้อมูลข่าวสาร", "ข้อมูลข่าวสาร"
    ]
    words = text.split()

    def find_positions(words, keywords):
        positions = []
        joined_text = "".join(words)
        for kw in keywords:
            start = 0
            while True:
                idx = joined_text.find(kw.replace(" ", ""), start)
                if idx == -1:
                    break
                positions.append(len(joined_text[:idx].split()))
                start = idx + len(kw.replace(" ", ""))
        return positions

    media_positions = find_positions(words, media_use_keywords)
    hidden_positions = find_positions(words, hidden_intent_keywords)
    effect_positions = find_positions(words, effect_keywords)
    # ตำแหน่ง media ก่อน hidden หรือ effect (ตามแบบเดิม)
    media_before_hidden = any((0 < h - m <= 5) for m in media_positions for h in hidden_positions)
    media_before_effect = any((0 < e - m <= 5) for m in media_positions for e in effect_positions)

    # ตรวจพบกลุ่ม keyword
    found_hidden_intent = find_keywords_list(text_norm, hidden_intent_keywords)
    found_effect = find_keywords_list(text_norm, effect_keywords)
    found_credibility = find_keywords_list(text_norm, credibility_keywords)

    # ต้องเจอทั้ง hidden_intent ผลกระทบ และ credibility ครบทั้ง 3 อย่าง
    score = 1 if (found_hidden_intent and found_effect and found_credibility) else 0
    return score

def evaluate_mind_score(answer_text):
    score1 = score_group_1(answer_text)
    score2 = score_group_2(answer_text)
    score3 = score_group_3(answer_text)
    score4 = score_group_4(answer_text)
    total_score = score1 + score2 + score3 + score4

    result = {
        "ใจความที่ 1": score1,
        "ใจความที่ 2": score2,
        "ใจความที่ 3": score3,
        "ใจความที่ 4": score4,
        "คะแนนรวมใจความ": total_score
    }

    return result

# -----------------------------
# Helper: ข้ามคำภาษาอังกฤษ/ตัวเลข
# -----------------------------
def is_english_or_number(word: str) -> bool:
    """
    คืน True ถ้า word เป็นภาษาอังกฤษหรือตัวเลข (หรือประกอบด้วยสัญลักษณ์ ASCII บางชนิด)
    """
    w = (word or "").strip()
    if not w:
        return False
    # อนุญาต A-Z a-z 0-9 และ . , ( ) - _ /
    return bool(re.fullmatch(r"[A-Za-z0-9\.\,\-\(\)_/]+", w))

# -----------------------------
# ตรวจ common misspellings (จากข้อความดิบ)
# -----------------------------
def check_common_misspellings_before_tokenize(text: str, misspelling_dict: dict):
    """
    text : ข้อความดิบ (ยังไม่ tokenize)
    misspelling_dict : dict เช่น { "ผิพท์": "พิมพ์", ... }
    คืน list ของ dict ที่มี keys: 'word' (wrong), 'index' (ตำแหน่งในข้อความดิบ), 'right' (คำถูก)
    """
    errors = []
    if not misspelling_dict:
        return errors
    for wrong, right in misspelling_dict.items():
        if wrong in text:
            for m in re.finditer(re.escape(wrong), text):
                errors.append({
                    "word": wrong,
                    "index": m.start(),
                    "right": right
                })
    return errors

# -----------------------------
# ตรวจ loanwords ก่อน tokenize (ใช้กับ tokens)
# -----------------------------
def check_loanword_before_tokenize(tokens, whitelist):
    """
    tokens : list ของ token (ตัดแล้ว)
    whitelist : set/list ของคำทับศัพท์ภาษาไทย (ไทยเขียนไม่ผิด)
    คืน list ของ dict: {'word': token, 'index': position, 'suggestions': [best_match]}
    """
    mistakes = []
    wl_list = list(whitelist) if whitelist else []
    for i, w in enumerate(tokens):
        if not w or is_english_or_number(w):
            continue
        # หา match ใกล้เคียงจาก whitelist
        matches = difflib.get_close_matches(w, wl_list, n=1, cutoff=0.7)
        if matches and w not in whitelist:
            mistakes.append({
                "word": w,
                "index": i,
                "suggestions": [matches[0]]
            })
    return mistakes

#ตรวจการฉีกคำ
def check_linebreak_issue(prev_line_tokens, next_line_tokens, max_words=3):
    last_word = prev_line_tokens[-1]
    first_word = next_line_tokens[0]
    if last_word.endswith('-') or first_word.startswith('-'):
        return False, None, None, None
    for prev_n in range(1, min(max_words, len(prev_line_tokens)) + 1):
        prev_part = ''.join(prev_line_tokens[-prev_n:])
        for next_n in range(1, min(max_words, len(next_line_tokens)) + 1):
            next_part = ''.join(next_line_tokens[:next_n])
            combined = normalize(prev_part + next_part)
            if (
                (' ' not in combined)
                and (combined not in splitable_phrases)
                and (
                    (combined in strict_not_split_words) or (
                        (combined in thai_dict)
                        and (len(word_tokenize(combined, engine='newmm')) == 1)
                    )
                )
            ):
                return True, prev_part, next_part, combined
    return False, None, None, None

#วนตรวจทั้งข้อความทีละบรรทัด
def analyze_linebreak_issues(text):
    lines = text.strip().splitlines()
    issues = []
    for i in range(len(lines) - 1):
        prev_line = lines[i].strip()
        next_line = lines[i + 1].strip()
        prev_tokens = word_tokenize(prev_line)
        next_tokens = word_tokenize(next_line)
        if not prev_tokens or not next_tokens:
            continue
        issue, prev_part, next_part, combined = check_linebreak_issue(prev_tokens, next_tokens)
        if issue:
            issues.append({
                'line_before': prev_line,
                'line_after': next_line,
                'prev_part': prev_part,
                'next_part': next_part,
                'combined': combined,
                'pos_in_text': (i, len(prev_tokens))
            })
    return issues

#รวมข้อความหรือคำที่ถูกตัดข้ามบรรทัด
def merge_linebreak_words(text, linebreak_issues):
    lines = text.splitlines()
    for issue in reversed(linebreak_issues):
        i, _ = issue['pos_in_text']
        lines[i] = lines[i].rstrip() + issue['combined'] + lines[i+1].lstrip()[len(issue['next_part']):]
        lines.pop(i+1)
    return "\n".join(lines)

#ตรวจการสสะกดคำ pythainlp + longdo
def pythainlp_spellcheck(tokens, pos_tags, dict_words=None, ignore_words=None):
    if dict_words is None:
        dict_words = thai_dict
    if ignore_words is None:
        ignore_words = set()
    misspelled = []
    for i, w in enumerate(tokens):
        if not w.strip() or w in dict_words or w in ignore_words or len(w) == 1 or 'ๆ' in w:
            continue
        misspelled.append({
            'word': w,
            'pos': pos_tags[i][1] if i < len(pos_tags) else None,
            'index': i
        })
    return misspelled

def longdo_spellcheck_batch(words):
    results = {}
    if not words:
        return results
    try:
        headers = {'Content-Type': 'application/json'}
        payload = {"key": API_KEY, "text": "\n".join(words)}
        response = requests.post(API_URL, headers=headers, json=payload, timeout=6)
        if response.status_code == 200:
            result = response.json()
            for e in result.get("result", []):
                if e.get("suggestions"):
                    results[e["word"]] = e["suggestions"]
    except Exception as e:
        print(f"Exception calling longdo: {e}")
    return results

#ตรวจการสะกดคำของคำทับศัพท์
def check_loanword_spelling(tokens,loanwords_whitelist):
    mistakes = []
    for tok in tokens:
        # Find close matches with a lower cutoff for loanwords
        matches = difflib.get_close_matches(tok, list(loanwords_whitelist), n=1, cutoff=0.7) # Lowered cutoff
        if matches and tok not in loanwords_whitelist:
            mistakes.append({'found': tok, 'should_be': matches[0]})
    return mistakes

#ตรวจการใช้เครื่องหมายที่ไม่อนุญาต
def find_unallowed_punctuations(text):
    pattern = f"[^{''.join(re.escape(p) for p in allowed_punctuations)}a-zA-Z0-9ก-๙\\s]"
    return set(re.findall(pattern, text))

#ใช้แยกไม้ยมกออกจากคำที่ติดกัน
def separate_maiyamok(text):
    return re.sub(r'(\S+?)ๆ', r'\1 ๆ', text)
#ตรวจการใช้ไม้ยมก
def analyze_maiyamok(tokens, pos_tags):
    results = []
    found_invalid = False
    VALID_POS = {'NCMN', 'NNP', 'VACT', 'VNIR', 'CLFV', 'ADVN', 'ADVI', 'ADVP', 'PRP', 'ADV'}
    for i, token in enumerate(tokens):
        if token == 'ๆ':
            prev_idx = i - 1
            prev_word = tokens[prev_idx] if prev_idx >= 0 else None
            prev_tag = pos_tags[prev_idx][1] if prev_idx >= 0 else None
            if prev_word is None or prev_word == 'ๆ':
                verdict = "❌ ไม้ยมกไม่ควรขึ้นต้นประโยค/คำ"
            elif prev_word in forbid_list:
                verdict = '❌ ไม่ควรใช้ไม้ยมกกับคำนี้'
            elif (prev_tag in VALID_POS) or (prev_word in allow_list):
                verdict = '✅ ถูกต้อง (ใช้ไม้ยมกซ้ำคำได้)'
            else:
                verdict = '❌ ไม่ควรใช้ไม้ยมok นอกจากกับคำนาม/กริยา/วิเศษณ์'
            context = tokens[max(0, i-2):min(len(tokens), i+3)]
            results.append({
                'คำก่อนไม้ยมก': prev_word or '',
                'POS คำก่อน': prev_tag or '',
                'บริบท': ' '.join(context),
                'สถานะ': verdict
            })
            if verdict.startswith('❌'):
                found_invalid = True
    return results, found_invalid

#ตรวจการแยกคำ
def detect_split_errors(tokens, custom_words=None):
    check_dict = set(thai_words()).union(custom_words or [])
    check_dict = {w for w in check_dict if (' ' not in w) and w.strip()}
    errors = []
    for i in range(len(tokens) - 1):
        combined = tokens[i] + tokens[i + 1]
        if (' ' not in combined) and (combined in check_dict) and (combined not in splitable_phrases):
            errors.append({
                "split_pair": (tokens[i], tokens[i+1]),
                "suggested": combined
            })
    return errors

def evaluate_text(text):
    # -----------------------------
    # จัดการตัดบรรทัด
    # -----------------------------
    linebreak_issues = analyze_linebreak_issues(text)
    corrected_text = merge_linebreak_words(text, linebreak_issues)

    # tokenize
    tokens = word_tokenize(corrected_text, engine='newmm', keep_whitespace=False)
    pos_tags = pos_tag(tokens, corpus='orchid')

    # ✅ 1) ตรวจ spelling ด้วย PyThaiNLP
    pythai_errors = pythainlp_spellcheck(tokens, pos_tags, dict_words=thai_dict, ignore_words=custom_words)

    # ✅ 2) ตรวจ Longdo (batch)
    wrong_words = [e['word'] for e in pythai_errors]
    longdo_results = longdo_spellcheck_batch(wrong_words)
    longdo_errors = [
        {**e, 'suggestions': longdo_results.get(e['word'], [])}
        for e in pythai_errors if e['word'] in longdo_results
    ]

    # ✅ 3) ตรวจ common misspellings จากข้อความดิบ
    json_misspells = check_common_misspellings_before_tokenize(corrected_text, COMMON_MISSPELLINGS)

    # ✅ 4) ตรวจ loanwords
    loanword_errors = check_loanword_before_tokenize(tokens, loanwords_whitelist)

    # ✅ รวม spelling errors ทั้งหมด
    all_spelling_errors = longdo_errors + [
        {
            "word": e["word"],
            "pos": None,
            "index": e["index"],
            "suggestions": [e["right"]],
        }
        for e in json_misspells
    ] + loanword_errors

    # ✅ ตรวจ punctuation, maiyamok, split word
    punct_errors = find_unallowed_punctuations(corrected_text)
    maiyamok_results, has_wrong_maiyamok = analyze_maiyamok(tokens, pos_tags)
    split_errors = detect_split_errors(tokens, custom_words=custom_words)

    # ✅ รวมผล errors
    error_counts = {
        "spelling": len(all_spelling_errors),
        "linebreak": len(linebreak_issues),
        "split": len(split_errors),
        "punct": len(punct_errors),
        "maiyamok": sum(1 for r in maiyamok_results if r['สถานะ'].startswith('❌'))
    }

    # ✅ สร้าง reasons
    reasons = []
    if error_counts["linebreak"]:
        details = [f"{issue['prev_part']} + {issue['next_part']} → {issue['combined']}" for issue in linebreak_issues]
        reasons.append("พบการฉีกคำข้ามบรรทัด: " + "; ".join(details))
    if error_counts["split"]:
        details = [f"{e['split_pair'][0]} + {e['split_pair'][1]} → {e['suggested']}" for e in split_errors]
        reasons.append("พบการแยกคำผิด: " + "; ".join(details))
    if error_counts["spelling"]:
        error_words = [
            f"{e['word']} (แนะนำ: {', '.join(e.get('suggestions', []))})"
            for e in all_spelling_errors
        ]
        reasons.append(f"ตรวจเจอคำสะกดผิดหรือทับศัพท์ผิด: {', '.join(error_words)}")
    if error_counts["punct"]:
        reasons.append(f"ใช้เครื่องหมายที่ไม่อนุญาต: {', '.join(punct_errors)}")
    if error_counts["maiyamok"]:
        wrong_desc = [x for x in maiyamok_results if x['สถานะ'].startswith('❌')]
        texts = [f"{x['คำก่อนไม้ยมก']}: {x['สถานะ']}" for x in wrong_desc]
        reasons.append("ใช้ไม้ยมกผิด: " + '; '.join(texts))
    if not reasons:
        reasons.append("ไม่มีปัญหา")

    # ✅ การให้คะแนน
    if sum(error_counts.values()) == 0:
        score = 1.0
    elif sum(c > 0 for c in error_counts.values()) == 1 and max(error_counts.values()) >= 2:
        score = 0.0
    elif sum(c > 0 for c in error_counts.values()) == 1:
        score = 0.5
    else:
        score = 0.0

    return {
        'linebreak_issues': linebreak_issues,
        'spelling_errors': all_spelling_errors,
        'loanword_spell_errors': loanword_errors,
        'punctuation_errors': list(punct_errors),
        'maiyamok_results': maiyamok_results,
        'split_errors': split_errors,
        'reasons': reasons,
        'score': score
    }


# ==========================
# S2---ฟังก์ชันตรวจเรียงลำดับ/เชื่อมโยงความคิด
# ==========================




#---------S3 ความถูกต้องตามหลักการเขียนย่อความ-------------
# ---------- ตั้งค่า ----------
TNER_URL = 'https://api.aiforthai.in.th/tner'
#AIFORTHAI_URL = "https://api.aiforthai.in.th/qaiapp"


# ---------- โหลด Dataset ----------
examples_df = pd.read_csv(r'D:\project1\example_dialect.csv')
pronouns_df = pd.read_csv(r'D:\project1\personal_pronoun_dataset1 (1).csv')

example_phrases = examples_df['local_word'].dropna().tolist()
pronouns_1 = pronouns_df['personal pronoun 1'].dropna().tolist()
pronouns_2 = pronouns_df['personal pronoun 2'].dropna().tolist()
pronouns_1_2 = pronouns_1 + pronouns_2

# ---------- บทความอ้างอิง ----------
reference_text = """
สื่อสังคม (Social Media) หรือที่คนทั่วไปเรียกว่า สื่อออนไลน์ หรือ สื่อสังคม ออนไลน์ นั้น เป็นสื่อหรือช่องทางที่แพร่กระจายข้อมูลข่าวสารในรูปแบบต่างๆ ได้อย่างรวดเร็วไปยังผู้คนที่อยู่ทั่วทุกมุมโลกที่สัญญาณโทรศัพท์เข้าถึง เช่น การนําเสนอข้อดีนานาประการของสินค้าชั้นนํา สินค้าพื้นเมืองให้เข้าถึงผู้ซื้อได้
ทั่วโลก การนําเสนอข้อเท็จจริงของข่าวสารอย่างตรงไปตรงมา การเผยแพร่ งานเขียนคุณภาพบนโลกออนไลน์แทนการเข้าสํานักพิมพ์ เป็นต้น จึงกล่าวได้ว่า เราสามารถใช้สื่อสังคมออนไลน์ค้นหาและรับข้อมูลข่าวสารที่มีประโยชน์ได้เป็นอย่างดี
  อย่างไรก็ตาม หากใช้สื่อสังคมออนไลน์อย่างไม่ระมัดระวัง หรือขาดความรับผืดชอบต่อสังคมส่วนรวม ไม่ว่าจะเป็นการเขียนแสดงความคิดเห็นวิพากษ์วิจารณ์ผู้อื่นในทางเสียหาย การนำเสนอผลงานที่มีเนื้อหาล่อแหลมหรือชักจูงผู้รับสารไปในทางไม่เหมาะสม หรือการสร้างกลุ่มเฉพาะที่ขัดต่อศีลธรรมอันดีของสังคมตลอดจนใช้เป็นช่องทางในการกระทำผิดกฎหมายทั้งการพนัน การขายของ
ผิดกฎหมาย เป็นต้น การใช้สื่อสังคมออนไลน์ในลักษณะดังกล่าวจึงเป็นการใช้ที่เป็นโทษแก่สังคม
	ปัจจุบันผู้คนจํานวนไม่น้อยนิยมใช้สื่อสังคมออนไลน์เป็นช่องทางในการทํา การตลาดทั้งในทางธุรกิจ สังคม และการเมือง จนได้ผลดีแบบก้าวกระโดด ทั้งนี้ เพราะสามารถเข้าถึงกลุ่มคนทุกเพศ ทุกวัย และทุกสาขาอาชีพโดยไม่มีข้อจํากัดเรื่อง เวลาและสถานที่ กลุ่มต่างๆ ดังกล่าวจึงหันมาใช้สื่อสังคมออนไลน์เพื่อสร้างกระแสให้ เกิดความนิยมชมชอบในกิจการของตน ด้วยการโฆษณาชวนเชื่อทุกรูปแบบจนลูกค้า เกิดความหลงใหลข้อมูลข่าวสาร จนตกเป็นเหยื่ออย่างไม่รู้ตัว เราจึงควรแก้ปัญหา การตกเป็นเหยื่อทางการตลาดของกลุ่มมิจฉาชีพด้วยการเร่งสร้างภูมิคุ้มกันรู้ทันสื่อไม่ตกเป็นเหยื่อทางการตลาดโดยเร็ว
	แม้ว่าจะมีการใช้สื่อสังคมออนไลน์ในทางสร้างสรรค์สิ่งที่ดีให้แก่สังคม ตัวอย่างเช่น การเตือนภัยให้แก่คนในสังคมได้อย่างรวดเร็ว การส่งต่อข้อมูลข่าวสาร เพื่อระดมความช่วยเหลือให้แก่ผู้ที่กําลังเดือดร้อน เป็นต้น แต่หลายครั้งคนในสังคมก็ อาจรู้สึกไม่มั่นใจเมื่อพบว่าตนเองถูกหลอกลวงจากคนบางกลุ่มที่ใช้สื่อสังคมออนไลน์
เป็นพื้นที่แสวงหาผลประโยชน์ส่วนตัว จนทําให้เกิดความเข้าใจผิดและสร้างความ เสื่อมเสียให้แก่ผู้อื่น ดังนั้นการใช้สื่อสังคมออนไลน์ด้วยเจตนาแอบแฝงจึงมีผลกระทบต่อความน่าเชื่อถือของข้อมูลข่าวสารโดยตรง
"""

# ---------- ฟังก์ชันตรวจหลักการย่อความ ----------
def call_tner(text):
    headers = {'Apikey': API_KEY}
    data = {'text': text}
    try:
        resp = requests.post(TNER_URL, headers=headers, data=data, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        print(f"TNER API error: {e}")
    return None

def check_summary_similarity(student_answer, reference_text, threshold=0.8):
    embeddings = model.encode([student_answer, reference_text])
    sim = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return sim >= threshold, sim

def check_examples(student_answer, example_phrases):
    return not any(phrase in student_answer for phrase in example_phrases)

def check_pronouns(student_answer, pronouns_list):
    words = word_tokenize(student_answer, engine='newmm')
    return not any(p in words for p in pronouns_list)

def check_abbreviations(student_answer):
    pattern = r'\b(?:[ก-ฮA-Za-z]\.){2,}'
    if re.search(pattern, student_answer):
        return False
    tner_result = call_tner(student_answer)
    if tner_result:
        for item in tner_result.get('entities', []):
            if item['type'] in ['ABB_DES', 'ABB_TTL', 'ABB_ORG', 'ABB_LOC', 'ABB']:
                return False
    return True

def check_title(student_answer, forbidden_title="การใช้สื่อสังคมออนไลน์"):
    return forbidden_title not in student_answer

def validate_student_answer(student_answer):
    sim_pass, sim_score = check_summary_similarity(student_answer, reference_text)
    results = {
        "summary_similarity": sim_pass,
        "similarity_score": round(sim_score, 3),
        "no_example": check_examples(student_answer, example_phrases),
        "no_pronouns": check_pronouns(student_answer, pronouns_1_2),
        "no_abbreviations": check_abbreviations(student_answer),
        "no_title": check_title(student_answer),
    }
    errors = [k for k, v in results.items() if k != "similarity_score" and not v]
    score = 1 if len(errors) == 0 else 0
    return score, errors, results

#----------S6 การใช้ประโยค------------
# ---------------- Typhoon API ----------------
client = openai.OpenAI(
    api_key="sk-3u6WAA0DwMjJoJ2xDDxFy2ecuZDKTjUF1mCOCXAJKSlR3Xqq",
    base_url="https://api.opentyphoon.ai/v1"
)

# ---------------- AI for Thai API ----------------
#aiforthai_url = "https://api.aiforthai.in.th/qaiapp"
#aiforthai_headers = {
#    'Content-Type': "application/json",
#    'apikey': "pHeDDSTgNpK4jLxoHXDQsdt3b9LC5yRL",
#}

# ---------------- ฟังก์ชัน Typhoon ----------------
def ask_typhoon(question, document):
    response = client.chat.completions.create(
        model="typhoon-v2.1-12b-instruct",
        messages=[
            {"role": "system", "content": "คุณคือผู้เชี่ยวชาญด้านภาษาไทย"},
            {"role": "user", "content": f"{question} จากประโยค:\n{document}"}
        ],
        temperature=0,
        max_tokens=1000
    )
    return response.choices[0].message.content.strip()

# ---------------- ฟังก์ชัน AI for Thai ----------------



#------------------S7 คำบอกข้อคิดเห็น --------------------
def evaluate_agreement_with_reference(answer: str, reference_text: str, threshold: float = 0.6) -> dict:
    """
    ตรวจคำบอกข้อคิดเห็น (เห็นด้วย/ไม่เห็นด้วย) + cosine similarity กับ reference_text
    ใช้ตรวจ essay_analysis
    """
    found = None
    if "ไม่เห็นด้วย" in answer:
        found = "ไม่เห็นด้วย"
    elif "เห็นด้วย" in answer:
        found = "เห็นด้วย"

    emb_answer = model.encode(answer, convert_to_tensor=True)
    emb_ref = model.encode(reference_text, convert_to_tensor=True)
    cosine_score = float(util.cos_sim(emb_answer, emb_ref)[0][0].item())

    if not found and cosine_score < threshold:
        return {
            "cosine_similarity": round(cosine_score, 3),
            "found_word": "ไม่พบ",
            "score": 0,
            "message": "ไม่มีคำบอกข้อคิดเห็น และ cosine < threshold, ไม่ตรวจทั้งข้อ"
        }

    return {
        "cosine_similarity": round(cosine_score, 3),
        "found_word": found if found else "ไม่พบ",
        "score": 1 if found else 0.5,
        "message": "ตรวจผ่าน"
    }

#------------------S9 การเรียงลำดับ --------------------


#------------------S10 ความถูกต้องตามหลักการเขียนแสดงความคิดเห็น --------------------
TNER_API_KEY = 'pHeDDSTgNpK4jLxoHXDQsdt3b9LC5yRL'
CYBERBULLY_API_KEY = 'pHeDDSTgNpK4jLxoHXDQsdt3b9LC5yRL'

personal_pronoun_1 = {"หนู", "ข้า", "กู"}
personal_pronoun_2 = {"คุณ", "แก", "เธอ", "ตัวเอง", "เอ็ง", "มึง"}
all_personal_pronouns = personal_pronoun_1.union(personal_pronoun_2)

def check_named_entities(text):
    url = "https://api.aiforthai.in.th/tner"
    headers = {"Apikey": TNER_API_KEY}
    data = {"text": text}
    try:
        response = requests.post(url, headers=headers, data=data, timeout=8)
        if response.status_code == 200:
            ner_result = response.json()
            bad_tags = {'ABB_DES', 'ABB_TTL', 'ABB_ORG', 'ABB_LOC', 'ABB'}
            bad_entities = [ent['word'] for ent in ner_result.get("entities", []) if ent['tag'] in bad_tags]
            if bad_entities:
                return True, bad_entities
    except:
        pass
    return False, []

def check_cyberbully(text):
    url = "https://api.aiforthai.in.th/cyberbully"
    headers = {"Apikey": CYBERBULLY_API_KEY}
    data = {"text": text}
    try:
        response = requests.post(url, headers=headers, data=data, timeout=8)
        if response.status_code == 200:
            result = response.json()
            if result.get("bully", "no") == "yes":
                bully_words = result.get("bully_words") or result.get("bully_phrases") or [text]
                return True, bully_words
    except:
        pass
    return False, []

def check_personal_pronouns(text):
    tokens = word_tokenize(text, engine="newmm")
    found_pronouns = [token for token in tokens if token in all_personal_pronouns]
    if found_pronouns:
        return True, found_pronouns
    return False, []

def evaluate_comment_validity(text):
    """
    ตรวจความถูกต้องของการแสดงความคิดเห็น
    - ห้ามมีชื่อเฉพาะ
    - ห้ามใช้คำ bully
    - ห้ามใช้สรรพนามบุรุษที่ 1/2
    """
    mistakes = []
    mistake_count = 0

    ne_flag, ne_words = check_named_entities(text)
    if ne_flag:
        mistake_count += 1
        mistakes.append(f"มีชื่อเฉพาะ/ตัวย่อ: {', '.join(ne_words)}")

    bully_flag, bully_words = check_cyberbully(text)
    if bully_flag:
        mistake_count += 1
        mistakes.append(f"ข้อความลักษณะ Cyberbully: {', '.join(bully_words)}")

    pronoun_flag, pronouns = check_personal_pronouns(text)
    if pronoun_flag:
        mistake_count += 1
        mistakes.append(f"ใช้สรรพนามบุรุษที่ 1 หรือ 2: {', '.join(pronouns)}")

    if mistake_count == 0:
        score = 1
    elif mistake_count == 1:
        score = 0.5
    else:
        score = 0

    return {
        "score": score,
        "details": mistakes if mistakes else ["ไม่มีข้อผิดพลาด"]
    }


# ✅ -----------------------------
# ฟังก์ชันหลัก เรียกจาก FastAPI
# ✅ -----------------------------
def convert_numpy_to_python(obj):
    """แปลงค่า numpy หรือ set ให้ JSON-safe"""
    if isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(x) for x in obj]
    elif isinstance(obj, set):
        return [convert_numpy_to_python(x) for x in obj]  # set → list
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj

# ==========================
# ฟังก์ชันหลัก: ตรวจใจความ + สะกดคำ + เรียงลำดับ/เชื่อมโยง + การใช้ประโยค
# ==========================
def evaluate_single_answer(answer_text, essay_analysis):
    # ---------------------------
    # ✅ ข้อที่ 1 : ตรวจจาก answer_text
    # ---------------------------

    # 1) ใจความสำคัญ
    student_emb = model.encode(answer_text, convert_to_tensor=True)
    core_sentences = [
        "สื่อสังคมหรือสื่อออนไลน์หรือสื่อสังคมออนไลน์เป็นช่องทางที่ใช้ในการเผยแพร่หรือค้นหาหรือรับข้อมูลข่าวสาร",
        "การใช้สื่อสังคมหรือสื่อออนไลน์หรือสื่อสังคมออนไลน์อย่างไม่ระมัดระวังหรือขาดความรับผิดชอบจะเกิดโทษหรือผลเสียหรือข้อเสียหรือผลกระทบหรือสิ่งไม่ดี",
        "ผู้ใช้ต้องรู้ทันหรือรู้เท่าทันสื่อสังคมออนไลน์",
        "การใช้สื่อสังคมหรือสื่อออนไลน์หรือสื่อสังคมออนไลน์ด้วยเจตนาแอบแฝงมีผลกระทบต่อความน่าเชื่อถือของข้อมูลข่าวสาร"
    ]
    core_embs = model.encode(core_sentences, convert_to_tensor=True)
    cosine_scores = util.cos_sim(student_emb, core_embs)[0]
    best_score = float(cosine_scores.max().item())

    mind_score = evaluate_mind_score(answer_text)
    mind_total = int(mind_score.get("คะแนนรวมใจความ ", 0))

    # ถ้าใจความเป็น 0 หรือ cosine ต่ำกว่า 0.6 → ตัดจบ
    if mind_total == 0 or best_score < 0.6:
        mind_score = {
            "cosine_similarity": round(best_score, 3),
            "ใจความที่ 1": 0,
            "ใจความที่ 2": 0,
            "ใจความที่ 3": 0,
            "ใจความที่ 4": 0,
            "คะแนนรวม": 0,
            "message": "ใจความต่ำกว่ามาตรฐาน → ข้อที่ 1 = 0"
        }
        mind_total = 0
        # กำหนดค่าทุกส่วนของข้อที่ 1 เป็น 0 โดยไม่ตรวจต่อ
        ordering1_score, ordering1_details = 0, {}
        summary1_score, summary1_details = 0, {}
        spelling_score, spelling_res = 0, {"reasons": []}
        score_s6, s6_result = 0, {}
        total_score1 = 0
    else:
        # 2) เรียงลำดับความคิด


        # 3) ความถูกต้องตามหลักการย่อความ
        summary1_score, summary1_err, summary1_details = validate_student_answer(answer_text)
        summary1_score = int(summary1_score)
        summary1_details = convert_numpy_to_python(summary1_details)

        # 4) การสะกดคำ
        spelling_res = evaluate_text(answer_text)
        spelling_score = float(spelling_res.get("score", 0))
        spelling_reason = str(spelling_res.get("reasons", ""))

        # 5) การใช้ประโยค (S6)

    

    # ---------------------------
    # ✅ ข้อที่ 2 : ตรวจจาก essay_analysis
    # ---------------------------

    # 1) คำบอกข้อคิดเห็น (เห็นด้วย/ไม่เห็นด้วย)
    agreement_result = evaluate_agreement_with_reference(essay_analysis, reference_text)
    agreement_score = agreement_result.get("score", 0)

    # ถ้าไม่มีคำบอก + cosine สูง → หยุดตรวจข้อที่ 2
    if agreement_score == 0 and "ไม่ตรวจ" in agreement_result.get("message",""):
        total_score1 = mind_total + ordering1_score + summary1_score + spelling_score + score_s6
        return convert_numpy_to_python({
            "ข้อที่ 1": {
                "ใจความสำคัญ (4 คะแนน)": mind_score,
                "เรียงลำดับ (2 คะแนน)": {"score": ordering1_score, "details": ordering1_details},
                "ความถูกต้องย่อความ (1 คะแนน)": {"score": summary1_score, "details": summary1_details},
                "การสะกดคำ (1 คะแนน)": {"score": spelling_score, "details": spelling_reason},
                "การใช้ประโยค (1 คะแนน)": s6_result,
                "คะแนนรวมข้อที่ 1": total_score1
            },
            "ข้อที่ 2": {
                "คำบอกข้อคิดเห็น": agreement_result,
                "message": "ไม่มีคำบอกข้อคิดเห็นและ cosine สูง → ข้อที่ 2 = 0 (ไม่ตรวจต่อ)"
            },
            "คะแนนรวมทั้งหมด": total_score1  # ไม่มีคะแนนข้อที่ 2
        })

    # 2) เรียงลำดับความคิด


    # 3) ความถูกต้องตามหลักการแสดงความคิดเห็น
    summary2_score, summary2_err, summary2_details = validate_student_answer(essay_analysis)
    summary2_score = int(summary2_score)
    summary2_details = convert_numpy_to_python(summary2_details)

    # ---------------------------
    # ✅ รวมคะแนนทั้งหมด
    # ---------------------------
    total_score1 = mind_total + summary1_score + spelling_score
    total_score2 = agreement_score  + summary2_score
    total_all = total_score1 + total_score2

    # ---------------------------
    # ✅ คืนค่า JSON-safe
    # ---------------------------
    return convert_numpy_to_python({
        # -------- ข้อที่ 1 --------
        "ข้อที่ 1 - ใจความสำคัญ": {
            **mind_score,
            "คะแนนรวม": mind_total
        },
        "ข้อที่ 1 - ความถูกต้องตามหลักการเขียนย่อความ": {
            "score": summary1_score,
            **summary1_details
        },
        "ข้อที่ 1 - การสะกดคำ": {
            "score": spelling_score,
            "reasons": spelling_res.get("reasons", [])
        },

        # -------- ข้อที่ 2 --------
        "ข้อที่ 2 - คำบอกข้อคิดเห็น": agreement_result,
        "ข้อที่ 2 - ความถูกต้องตามหลักการแสดงความคิดเห็น": {
            "score": summary2_score,
            **summary2_details
        },
        "คะแนนรวมข้อที่ 1": total_score1,
        "คะแนนรวมข้อที่ 2": total_score2,

        # -------- รวมทั้งหมด --------
        "คะแนนรวมทั้งหมด (15 คะแนน)": total_all
    })
