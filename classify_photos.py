"""
Классификация грибов по фото через Ollama (LLaVA).
Обрабатывает только посты, где вид не определён по тексту.
Берёт первое фото из поста.

Требует: ollama pull llava:13b
Запуск:  python classify_photos.py
"""

import os
import re
import json
import time
import base64
import requests
from tqdm import tqdm

INPUT_POSTS    = "data/raw_posts.json"
INPUT_SPECIES  = "data/posts_with_species.csv"
OUTPUT_CSV     = "data/photo_species.csv"
CHECKPOINT     = "data/photo_species_checkpoint.json"

OLLAMA_URL     = "http://localhost:11434/api/generate"
MODEL          = "llava:13b"

# Посты с этими словами — точно не отчёты, пропускаем фото-распознавание
SKIP_PHOTO_RE = re.compile("|".join([
    r"рецепт",                    # рецепты из грибов
    r"приготовл",                 # приготовление
    r"жарен|варен|тушен|маринов", # способы готовки
    r"суп\b|пирог|жульен|соус",  # блюда
    r"продаж|куплю|продам|цена",  # торговля
    r"отравлен|ядовит|опасн",     # предупреждения
    r"мухомор",                   # обычно не собирают
    r"реклам|подписк|розыгрыш",   # реклама
    r"карт[аыу]\s+грибн",         # карты грибных мест
    r"прогноз\s+погод",           # прогноз погоды
    r"стих|поэзи|цитат",          # стихи/цитаты
    r"конкурс|голосован",         # конкурсы
    r"правил[аоы]\s+(груп|сообщ)",# правила группы
    r"клещ|змея|змей|медвед",     # опасности леса
    r"ягод[аыу]|клюкв|брусник|черник|морошк", # ягоды
]), re.IGNORECASE)

PROMPT = """Look at this photo from a mushroom hunting community.

Answer in JSON format ONLY, no other text:
{"species": "...", "count": ..., "confidence": "..."}

species — one of: белый, лисичка, опёнок, маслёнок, подосиновик, подберёзовик, сморчок, сморчковая_шапочка, рыжик, груздь, моховик, козляк, сыроежка, волнушка, горькушка, другой, не_гриб, не_фото_грибов
count — estimated number of mushrooms visible (integer). If mushrooms are in a basket/bucket and hard to count exactly, estimate. If >50, write 50.
confidence — high, medium, low

If there are multiple species, list only the dominant one.
If the photo doesn't show mushrooms (landscape, person, food, etc.), use species "не_фото_грибов" and count 0."""


def download_photo(url: str, timeout: int = 15) -> bytes | None:
    """Скачивает фото по URL, возвращает bytes."""
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200 and len(resp.content) > 1000:
            return resp.content
        return None
    except Exception:
        return None


def ask_ollama(image_bytes: bytes) -> dict | None:
    """Отправляет фото в Ollama, возвращает распознанный вид."""
    img_b64 = base64.b64encode(image_bytes).decode()

    try:
        resp = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": PROMPT,
            "images": [img_b64],
            "stream": False,
            "options": {"temperature": 0.1},
        }, timeout=60)

        if resp.status_code != 200:
            return None

        text = resp.json().get("response", "").strip()

        # Парсим JSON из ответа
        # LLM может обернуть в ```json ... ``` или добавить текст
        import re
        json_match = re.search(r"\{[^}]+\}", text)
        if json_match:
            return json.loads(json_match.group())
        return None

    except Exception as e:
        print(f"  Ollama error: {e}")
        return None


def load_checkpoint() -> dict:
    """Загружает чекпоинт — уже обработанные посты."""
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(results: dict):
    """Сохраняет чекпоинт."""
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


def main():
    # Загружаем посты
    with open(INPUT_POSTS, encoding="utf-8") as f:
        posts = json.load(f)

    # Все посты с фото, кроме зимних (ноя–мар)
    from datetime import datetime
    WINTER_MONTHS = {11, 12, 1, 2, 3}

    to_process = []
    skipped_winter = 0
    skipped_text = 0
    for p in posts:
        pid = str(p["id"])
        urls = p.get("photo_urls", [])
        if not urls:
            continue
        # Пропускаем зимние месяцы
        post_month = datetime.strptime(p["date_posted"], "%Y-%m-%d").month
        if post_month in WINTER_MONTHS:
            skipped_winter += 1
            continue
        # Пропускаем нерелевантные по тексту (рецепты, ягоды, реклама...)
        text = p.get("text", "")
        if text and SKIP_PHOTO_RE.search(text):
            skipped_text += 1
            continue
        to_process.append({
            "id": pid,
            "url": urls[0],
            "n_photos": len(urls),
        })

    print(f"Постов с фото (апр-окт, релевантные): {len(to_process)}")
    print(f"Пропущено зимних: {skipped_winter}")
    print(f"Пропущено по тексту (рецепты, ягоды...): {skipped_text}")

    # Загружаем чекпоинт
    results = load_checkpoint()
    already_done = len(results)
    print(f"Уже обработано (чекпоинт): {already_done}")

    remaining = [p for p in to_process if p["id"] not in results]
    print(f"Осталось обработать: {len(remaining)}")

    if not remaining:
        print("Всё обработано!")
        save_results(results)
        return

    # Проверяем Ollama
    try:
        requests.get("http://localhost:11434/api/tags", timeout=5)
    except Exception:
        print("Ollama не запущена! Запусти: ollama serve")
        return

    # Обрабатываем
    errors = 0
    for i, post in enumerate(tqdm(remaining, desc="Распознаём фото")):
        # Скачиваем фото
        img = download_photo(post["url"])
        if img is None:
            results[post["id"]] = {"species": "ошибка", "error": "download_failed"}
            errors += 1
            continue

        # Отправляем в Ollama
        result = ask_ollama(img)
        if result is None:
            results[post["id"]] = {"species": "ошибка", "error": "ollama_failed"}
            errors += 1
        else:
            results[post["id"]] = result

        # Чекпоинт каждые 100 постов
        if (i + 1) % 100 == 0:
            save_checkpoint(results)
            recognized = sum(1 for r in results.values()
                           if isinstance(r, dict) and r.get("species") not in ("ошибка", "не_гриб", None))
            print(f"  [{i+1}/{len(remaining)}] распознано грибов: {recognized}, ошибок: {errors}")

    save_checkpoint(results)
    save_results(results)


def save_results(results: dict):
    """Сохраняет финальный CSV."""
    import csv
    os.makedirs("data", exist_ok=True)

    rows = []
    species_count = {}
    for pid, r in results.items():
        if not isinstance(r, dict):
            continue
        species = r.get("species", "ошибка")
        count = r.get("count", 0)
        try:
            count = int(count)
        except (ValueError, TypeError):
            count = 0
        confidence = r.get("confidence", "")
        rows.append({
            "id": pid,
            "photo_species": species,
            "photo_count": count,
            "photo_confidence": confidence,
        })
        species_count[species] = species_count.get(species, 0) + 1

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "photo_species", "photo_count", "photo_confidence"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nСохранено: {OUTPUT_CSV} ({len(rows)} записей)")
    print("\nВиды по фото:")
    for s, c in sorted(species_count.items(), key=lambda x: -x[1]):
        print(f"  {s:25s} {c:6d}")


if __name__ == "__main__":
    main()
