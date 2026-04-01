"""
Классификация грибов по фото через Ollama (LLaVA).
Обрабатывает все посты с фото (апр–окт), определяет вид и количество грибов.
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
MODEL          = "qwen2.5vl:7b"

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
    # Пустые походы
    r"ничего\s+не\s+наш",
    r"не\s+наш[ёел][иа]?\s+(ни|гриб)",
    r"пустой?\s+корзин",
    r"без\s+гриб",
    r"ни\s+одного\s+гриб",
    r"\bпролёт\b",
    r"\bвпустую\b",
    r"\bбезрезультат",
    # Поздравления
    r"с\s+днём?\s+рождени",
    r"день\s+рождени",
    r"\bдр\b",
    r"поздравля",
    r"с\s+новым\s+год",
    r"новогодн",
    r"8\s*марта",
    r"23\s*февраля",
    r"день\s+защитника",
]), re.IGNORECASE)

PROMPT = """What mushrooms are in this photo? Identify species and estimate total count.
Reply ONLY with JSON array:
[{"species": "name", "count": number}]
If no mushrooms visible: [{"species": "none", "count": 0}]
Counting: full basket=30-50, handful=5-10, individual=count exactly. Never write 0 if you see mushrooms."""


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

        # Парсим JSON массив из ответа
        # LLM может обернуть в ```json ... ``` или добавить текст
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                return parsed
        # Fallback: одиночный объект
        json_match = re.search(r"\{[^}]+\}", text)
        if json_match:
            return [json.loads(json_match.group())]
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
        if len(urls) >= 2:
            # Первое (красивые грибы) + последнее (полный улов) → возьмём max
            to_process.append({
                "id": pid,
                "urls": [urls[0], urls[-1]],
            })
        else:
            to_process.append({
                "id": pid,
                "urls": [urls[0]],
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

    # Параллельная обработка: скачиваем фото в N потоков, Ollama обрабатывает
    from concurrent.futures import ThreadPoolExecutor, as_completed
    N_WORKERS = 3  # параллельных запросов к Ollama

    # Маппинг латинских названий → русские
    SPECIES_MAP = {
        "boletus edulis": "белый", "boletus": "белый",
        "cantharellus cibarius": "лисичка", "cantharellus": "лисичка",
        "armillaria": "опёнок", "honey fungus": "опёнок",
        "suillus": "маслёнок", "suillus luteus": "маслёнок",
        "leccinum aurantiacum": "подосиновик", "leccinum": "подосиновик",
        "orange birch bolete": "подосиновик",
        "birch bolete": "подберёзовик",
        "morchella": "сморчок", "morel": "сморчок",
        "gyromitra": "строчок", "gyromitra esculenta": "строчок",
        "verpa": "сморчковая_шапочка", "verpa bohemica": "сморчковая_шапочка",
        "lactarius deliciosus": "рыжик", "lactarius": "рыжик",
        "lactifluus": "груздь",
        "xerocomus": "моховик", "xerocomellus": "моховик",
        "russula": "сыроежка",
        "none": "нет_грибов", "unknown": "другой",
    }

    def normalize_species(name):
        return SPECIES_MAP.get(name.lower().strip(), "другой")

    def process_one(post):
        all_results = {}  # species → sum count

        for url in post["urls"]:
            img = download_photo(url)
            if img is None:
                continue
            result = ask_ollama(img)
            if result is None:
                continue
            for item in result:
                sp = normalize_species(item.get("species", ""))
                cnt = item.get("count", 0)
                # Обработка диапазонов типа "30-50"
                if isinstance(cnt, str) and "-" in cnt:
                    parts = cnt.split("-")
                    try:
                        cnt = (int(parts[0]) + int(parts[1])) // 2
                    except (ValueError, IndexError):
                        cnt = 0
                try:
                    cnt = int(cnt)
                except (ValueError, TypeError):
                    cnt = 0
                cnt = min(cnt, 100)  # клип
                # Суммируем дубли (две корзины одного вида)
                all_results[sp] = all_results.get(sp, 0) + cnt

        if not all_results:
            return post["id"], [{"species": "ошибка", "count": 0}]

        merged = [{"species": sp, "count": cnt} for sp, cnt in all_results.items()]
        return post["id"], merged

    errors = 0
    pbar = tqdm(total=len(remaining), desc="Распознаём фото")

    with ThreadPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {}
        batch_start = 0
        BATCH = 100

        while batch_start < len(remaining):
            # Отправляем батч
            batch = remaining[batch_start:batch_start + BATCH]
            futures = {executor.submit(process_one, p): p for p in batch}

            for future in as_completed(futures):
                pid, result = future.result()
                results[pid] = result
                if isinstance(result, dict) and result.get("error"):
                    errors += 1
                pbar.update(1)

            batch_start += BATCH

            # Чекпоинт после каждого батча
            save_checkpoint(results)
            recognized = sum(
                1 for r in results.values()
                if isinstance(r, list) and any(
                    item.get("species") not in ("ошибка", "не_фото_грибов", None)
                    for item in r
                )
            )
            pbar.set_postfix(recognized=recognized, errors=errors)

    pbar.close()

    save_checkpoint(results)
    save_results(results)


def save_results(results: dict):
    """Сохраняет финальный CSV."""
    import csv
    os.makedirs("data", exist_ok=True)

    rows = []
    species_count = {}
    for pid, r in results.items():
        if not isinstance(r, list):
            # Старый формат (dict) или ошибка
            if isinstance(r, dict) and r.get("error"):
                rows.append({"id": pid, "photo_species": "ошибка", "photo_count": 0})
            continue
        for item in r:
            species = item.get("species", "ошибка")
            count = item.get("count", 0)
            try:
                count = int(count)
            except (ValueError, TypeError):
                count = 0
            rows.append({
                "id": pid,
                "photo_species": species,
                "photo_count": count,
            })
            species_count[species] = species_count.get(species, 0) + 1

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "photo_species", "photo_count"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nСохранено: {OUTPUT_CSV} ({len(rows)} записей)")
    print("\nВиды по фото:")
    for s, c in sorted(species_count.items(), key=lambda x: -x[1]):
        print(f"  {s:25s} {c:6d}")


if __name__ == "__main__":
    main()
