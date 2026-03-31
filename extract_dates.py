"""
Извлечение даты похода за грибами из текста поста.

Стратегия (каскад):
  1. Regex — быстро, бесплатно, покрывает ~80% случаев
  2. LLM (Claude) — для постов, где regex не нашёл дату

Результат сохраняется в data/posts_with_dates.csv
"""

import os
import re
import json
import csv
from datetime import datetime, date, timedelta
from typing import Optional
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

# ── Константы ─────────────────────────────────────────────────────────────────

MONTHS_RU = {
    "январ": 1, "феврал": 2, "март": 3, "апрел": 4,
    "май": 5, "маи": 5, "июн": 6, "июл": 7, "август": 8,
    "сентябр": 9, "октябр": 10, "ноябр": 11, "декабр": 12,
}

MONTHS_RU_PATTERN = (
    r"(январ[яеьи]?|феврал[яеьи]?|марта?|апрел[яеьи]?|"
    r"ма[йи]|июн[яеьи]?|июл[яеьи]?|август[ае]?|"
    r"сентябр[яеьи]?|октябр[яеьи]?|ноябр[яеьи]?|декабр[яеьи]?)"
)

INPUT_JSON = "data/raw_posts.json"
OUTPUT_CSV = "data/posts_with_dates.csv"

# Посты с этими словами пропускаем — не отчёты о походах
SKIP_PATTERNS = [
    r"архив",
    # День рождения
    r"с\s+днём?\s+рождени",
    r"день\s+рождени",
    r"\bдр\b",
    r"поздравля",
    # Новый год
    r"с\s+новым\s+год",
    r"новогодн",
    # 8 марта
    r"8\s*марта",
    r"восьм[ое]+\s+март",
    r"международн[ый]+\s+женск",
    r"с\s+праздником\s+весны",
    # 23 февраля
    r"23\s*февраля",
    r"двадцать\s+тр[её]тьего\s+февраля",
    r"день\s+защитника",
    r"день\s+защитник",
]
SKIP_RE = re.compile("|".join(SKIP_PATTERNS), re.IGNORECASE)

# ── Regex-парсинг ──────────────────────────────────────────────────────────────

def _month_num(word: str) -> Optional[int]:
    word = word.lower()
    for prefix, num in MONTHS_RU.items():
        if word.startswith(prefix):
            return num
    return None


def parse_date_regex(text: str, post_date: str) -> Optional[str]:
    """
    Возвращает дату в формате YYYY-MM-DD или None.
    post_date — дата публикации поста (YYYY-MM-DD), нужна для «вчера», «сегодня».
    """
    post_dt = datetime.strptime(post_date, "%Y-%m-%d").date()
    text_lower = text.lower()

    # 1a. DD.MM.YYYY или DD/MM/YYYY или DD-MM-YYYY (+ "г" на конце: "31.01.2026г")
    # Пробелы вокруг разделителей допускаются: "10. 09.2025", "22 08.2025"
    # \b перед числом не требуем — ловим даже слипшиеся с текстом ("попёрли25.08.2025")
    m = re.search(r"(\d{1,2})\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{2,4})г?", text)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        # Исключаем время (ЧЧ:ММ не попадёт сюда, но на всякий случай)
        if 1 <= month <= 12 and 1 <= day <= 31:
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                pass

    # 1b. Диапазон дат "27-28.01.26" или "27-28.01.2026" — берём первую дату
    m = re.search(r"(\d{1,2})-\d{1,2}\s*[.\-/]\s*(\d{1,2})\s*[.\-/]\s*(\d{2,4})г?", text)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        if 1 <= month <= 12 and 1 <= day <= 31:
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                pass

    # 1c. Склеенный формат DD.MMYY или DD.MMYYYY (без разделителя перед годом: "21.0126")
    m = re.search(r"\b(\d{1,2})\.(\d{2})(\d{2,4})\b", text)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        if 1 <= month <= 12 and 1 <= day <= 31:
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                pass

    # 1d. DD.MM или DD/MM без года — НЕ ловим время (ЧЧ.ММ где ЧЧ<=23 и ММ<=59)
    m = re.search(r"\b(\d{1,2})[./\\](\d{1,2})\b", text)
    if m:
        day, month = int(m.group(1)), int(m.group(2))
        # Исключаем время: если это выглядит как ЧЧ.ММ (часы ≤23, минуты ≤59, месяц >12)
        looks_like_time = (day <= 23 and month <= 59 and month > 12)
        if not looks_like_time and 1 <= month <= 12 and 1 <= day <= 31:
            year = post_dt.year
            try:
                d = date(year, month, day)
                if d > post_dt:
                    d = date(year - 1, month, day)
                return d.isoformat()
            except ValueError:
                pass

    # 2. DD месяц YYYY  или  DD месяц  (без года)
    pattern_dmy = rf"\b(\d{{1,2}})\s+{MONTHS_RU_PATTERN}(?:\s+(\d{{4}}))?"
    m = re.search(pattern_dmy, text_lower)
    if m:
        day = int(m.group(1))
        month = _month_num(m.group(2))
        year_str = m.group(3)
        year = int(year_str) if year_str else post_dt.year
        if month:
            try:
                d = date(year, month, day)
                # Если дата «в будущем» относительно поста — вероятно прошлый год
                if d > post_dt:
                    d = date(year - 1, month, day)
                return d.isoformat()
            except ValueError:
                pass

    # 3. месяц DD  (например "октября 5")
    pattern_mdy = rf"{MONTHS_RU_PATTERN}\s+(\d{{1,2}})(?:\s+(\d{{4}}))?"
    m = re.search(pattern_mdy, text_lower)
    if m:
        month = _month_num(m.group(1))
        day = int(m.group(2))
        year_str = m.group(3)
        year = int(year_str) if year_str else post_dt.year
        if month:
            try:
                d = date(year, month, day)
                if d > post_dt:
                    d = date(year - 1, month, day)
                return d.isoformat()
            except ValueError:
                pass

    # 4. DD MM.YYYY — пробел между днём и месяцем: "22 08.2025", "27 08.2024"
    m = re.search(r"\b(\d{1,2})\s+(\d{2})\.(\d{4})\b", text)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                pass

    # 5. DD MM YYYY — всё через пробел: "05 11 2025", "24 09 2025"
    m = re.search(r"\b(\d{1,2})\s+(\d{2})\s+(\d{4})\b", text)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= month <= 12 and 1 <= day <= 31:
            try:
                return date(year, month, day).isoformat()
            except ValueError:
                pass

    # 6. Диапазон с месяцем: "9 и 12 мая", "11 и 13 октября" — берём первую дату
    pattern_range = rf"\b(\d{{1,2}})\s+и\s+\d{{1,2}}\s+{MONTHS_RU_PATTERN}(?:\s+(\d{{4}}))?"
    m = re.search(pattern_range, text_lower)
    if m:
        day = int(m.group(1))
        month = _month_num(m.group(2))
        year_str = m.group(3)
        year = int(year_str) if year_str else post_dt.year
        if month:
            try:
                d = date(year, month, day)
                if d > post_dt:
                    d = date(year - 1, month, day)
                return d.isoformat()
            except ValueError:
                pass

    # 7. День недели: "в субботу", "в воскресенье" → ближайший такой день до даты поста
    WEEKDAYS_RU = {
        "понедельник": 0, "вторник": 1, "среду": 2, "среда": 2,
        "четверг": 3, "пятницу": 4, "пятница": 4,
        "субботу": 5, "суббота": 5, "воскресенье": 6,
    }
    m = re.search(r"\bв\s+(" + "|".join(WEEKDAYS_RU) + r")\b", text_lower)
    if m:
        target_wd = WEEKDAYS_RU[m.group(1)]
        days_back = (post_dt.weekday() - target_wd) % 7
        if days_back == 0:
            days_back = 7  # "в субботу" в субботний пост — скорее всего прошлая
        d = post_dt - timedelta(days=days_back)
        # Не берём если получилось больше 14 дней назад — слишком неточно
        if days_back <= 14:
            return d.isoformat()

    # 8. Относительные: сегодня / вчера / позавчера
    if re.search(r"\bсегодня\b", text_lower):
        return post_dt.isoformat()
    if re.search(r"\bвчера\b", text_lower):
        return (post_dt - timedelta(days=1)).isoformat()
    if re.search(r"\bпозавчера\b", text_lower):
        return (post_dt - timedelta(days=2)).isoformat()

    return None


# ── LLM-парсинг (Claude) ───────────────────────────────────────────────────────

_client = None

def _get_client():
    global _client
    if _client is None:
        import anthropic
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    return _client


def parse_date_llm(text: str, post_date: str) -> Optional[str]:
    """
    Спрашивает Claude, какого числа человек ходил за грибами.
    Возвращает дату YYYY-MM-DD или None.
    Используется только когда regex не справился.
    """
    client = _get_client()
    prompt = f"""Вот текст поста из группы грибников. Дата публикации поста: {post_date}.
Определи, какого числа автор ходил за грибами (дата похода, не дата публикации).
Ответь ТОЛЬКО датой в формате YYYY-MM-DD. Если дата не указана — ответь UNKNOWN.

Текст поста:
{text[:1000]}"""

    try:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=20,
            messages=[{"role": "user", "content": prompt}],
        )
        result = msg.content[0].text.strip()
        # Проверяем, что это валидная дата
        datetime.strptime(result, "%Y-%m-%d")
        return result
    except Exception:
        return None


# ── Основной pipeline ──────────────────────────────────────────────────────────

def process_posts(use_llm: bool = False):
    with open(INPUT_JSON, encoding="utf-8") as f:
        posts = json.load(f)

    print(f"Обрабатываем {len(posts)} постов...")

    results = []
    llm_count = 0
    no_date_count = 0
    skipped_count = 0

    for post in tqdm(posts, desc="Извлекаем даты"):
        post_date = post["date_posted"]
        text = post["text"]

        # Пропускаем посты без текста или нерелевантные (архив, ДР, поздравления)
        if not text.strip():
            source = "no_text"
            foray_date = None
        elif SKIP_RE.search(text):
            source = "skipped"
            foray_date = None
            skipped_count += 1
        else:
            foray_date = parse_date_regex(text, post_date)
            source = "regex" if foray_date else None

            if foray_date is None and use_llm:
                foray_date = parse_date_llm(text, post_date)
                if foray_date:
                    source = "llm"
                    llm_count += 1

            if foray_date is None:
                source = "not_found"
                no_date_count += 1

        results.append({
            "id": post["id"],
            "date_posted": post_date,
            "foray_date": foray_date,
            "date_source": source,
            "likes": post["likes"],
            "views": post.get("views", 0),
            "photos": post["photos"],
            "text_preview": text[:200].replace("\n", " "),
        })

    # Сохраняем
    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Статистика
    found = sum(1 for r in results if r["foray_date"])
    print(f"\n{'-'*40}")
    relevant = len(posts) - skipped_count
    print(f"Всего постов:          {len(posts)}")
    print(f"Пропущено (архив/ДР):  {skipped_count}")
    print(f"Дата найдена (regex):  {found - llm_count}")
    print(f"Дата найдена (LLM):    {llm_count}")
    print(f"Дата не найдена:       {no_date_count}")
    print(f"Покрытие (от релев.):  {found/relevant*100:.1f}%")
    print(f"Результат: {OUTPUT_CSV}")


if __name__ == "__main__":
    import sys
    use_llm = "--llm" in sys.argv
    if use_llm and not os.getenv("ANTHROPIC_API_KEY"):
        print("Для LLM нужен ANTHROPIC_API_KEY в .env")
        sys.exit(1)
    process_posts(use_llm=use_llm)
