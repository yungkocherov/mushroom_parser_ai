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

    # 1. DD.MM.YYYY или DD/MM/YYYY или DD-MM-YYYY
    m = re.search(r"\b(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})\b", text)
    if m:
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if year < 100:
            year += 2000
        try:
            return date(year, month, day).isoformat()
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

    # 4. Относительные: сегодня / вчера / позавчера
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

    for post in tqdm(posts, desc="Извлекаем даты"):
        post_date = post["date_posted"]
        text = post["text"]

        # Пропускаем посты без текста
        if not text.strip():
            source = "no_text"
            foray_date = None
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
    print(f"\n{'─'*40}")
    print(f"Всего постов:          {len(posts)}")
    print(f"Дата найдена (regex):  {found - llm_count}")
    print(f"Дата найдена (LLM):    {llm_count}")
    print(f"Дата не найдена:       {no_date_count}")
    print(f"Покрытие:              {found/len(posts)*100:.1f}%")
    print(f"Результат: {OUTPUT_CSV}")


if __name__ == "__main__":
    import sys
    use_llm = "--llm" in sys.argv
    if use_llm and not os.getenv("ANTHROPIC_API_KEY"):
        print("Для LLM нужен ANTHROPIC_API_KEY в .env")
        sys.exit(1)
    process_posts(use_llm=use_llm)
