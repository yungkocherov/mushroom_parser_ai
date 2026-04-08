"""
Классификация грибов по фото через LM Studio (vision-модель).
Обрабатывает все посты с фото (апр-окт), сохраняет сырые ответы модели.
Берёт первое + последнее фото поста.

Требует: LM Studio запущена с vision-моделью на localhost:1234
Запуск:  python classify_photos.py
"""

import os
import re
import json
import base64
import requests
from tqdm import tqdm

INPUT_POSTS       = None
NO_MUSHROOM_IDS   = None
OUTPUT_CSV        = None
CHECKPOINT        = None

API_URL           = None
MODEL             = None

# Посты с этими словами — точно не отчёты, пропускаем фото-распознавание
SKIP_PHOTO_RE = re.compile("|".join([
    r"рецепт",                    # рецепты из грибов
    r"приготовл",                 # приготовление
    r"жарен|варен|тушен|маринов", # способы готовки
    r"суп\b|пирог|жульен|соус",  # блюда
    r"продаж|куплю|продам|цена",  # торговля
    r"отравлен|ядовит|опасн",     # предупреждения
    r"реклам|подписк|розыгрыш",   # реклама
    r"карт[аыу]\s+грибн",         # карты грибных мест
    r"прогноз\s+погод",           # прогноз погоды
    r"стих|поэзи|цитат",          # стихи/цитаты
    r"конкурс|голосован",         # конкурсы
    r"правил[аоы]\s+(груп|сообщ)",# правила группы
    r"клещ|змея|змей|медвед",     # опасности леса
    r"ягод[аыу]|клюкв|брусник|черник|морошк", # ягоды
    r"рыбалк|рыб[аыу]|щук[аиу]|окун[ьяей]|удочк|спиннинг|улов[а-я]*\s+рыб|карас[ьяей]|лещ", # рыбалка
    # Пустые походы
    r"ничего\s+не\s+наш",
    r"не\s+наш[ёел][иа]?\s+(ни|гриб)",
    r"пустой?\s+корзин",
    r"без\s+гриб",
    r"ни\s+одного\s+гриб",
    r"\bпролёт\b",
    r"\bвпустую\b",
    r"\bбезрезультат",
    # Фотоохота и природа (не грибы)
    r"фотоохот",
    r"птиц[аыу]|пернат",
    r"закат[а-я]*|рассвет",
    r"\bцвет[ыуов][а-я]*\b(?!\s+волнушк)",  # цветы, но не волнушка
    r"весенние\s+(цветы|цветк)",
    r"ландыш|примула|мать-и-мачеха|мускари",
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

PROMPT = """Classify mushrooms in this photo into one of these groups and count them.
JSON only: [{"species":"group","count":N}]
No mushrooms: [{"species":"none","count":0}]
Basket=30-50, handful=5-10.

Groups:
- chanterelle (Cantharellus, any chanterelle species)
- bolete (Boletus, Leccinum, Suillus, Xerocomus - any tube mushroom with sponge under cap)
- morel (Morchella, Gyromitra, Verpa - wrinkled/brain-like cap, spring mushrooms)
- honey_fungus (Armillaria, Kuehneromyces - clusters on wood)
- other (any mushroom not matching above groups)"""


def download_photo(url: str, timeout: int = 15) -> bytes | None:
    """Скачивает фото по URL."""
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200 and len(resp.content) > 1000:
            return resp.content
        return None
    except Exception:
        return None


def ask_model(image_bytes: bytes) -> list | None:
    """Отправляет фото в vision-модель, возвращает список видов или None при ошибке."""
    img_b64 = base64.b64encode(image_bytes).decode()

    try:
        resp = requests.post(API_URL, json={
            "model": MODEL,
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}},
                {"type": "text", "text": PROMPT},
            ]}],
            "temperature": 0.1,
            "max_tokens": 200,
        }, timeout=60)

        if resp.status_code != 200:
            return None

        text = resp.json()["choices"][0]["message"]["content"].strip()

        # Фиксим невалидный JSON: "count": 30-50 → "count": "30-50"
        text = re.sub(r'"count"\s*:\s*(\d+)\s*-\s*(\d+)', r'"count": "\1-\2"', text)

        # Парсим JSON массив из ответа
        # LLM может обернуть в ```json ... ``` или добавить текст
        json_match = re.search(r"\[.*\]", text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
                if isinstance(parsed, list):
                    return parsed
            except json.JSONDecodeError:
                pass
        # Fallback: одиночный объект
        json_match = re.search(r"\{[^}]+\}", text)
        if json_match:
            try:
                return [json.loads(json_match.group())]
            except json.JSONDecodeError:
                pass
        return None

    except Exception as e:
        print(f"  Model error: {e}")
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


def main(city_config=None, app_config=None):
    global INPUT_POSTS, NO_MUSHROOM_IDS, OUTPUT_CSV, CHECKPOINT, API_URL, MODEL

    if city_config:
        INPUT_POSTS     = city_config.path("raw_posts.json")
        NO_MUSHROOM_IDS = city_config.path("no_mushroom_ids.json")
        OUTPUT_CSV      = city_config.path("photo_species.csv")
        CHECKPOINT      = city_config.path("photo_species_checkpoint.json")
    else:
        INPUT_POSTS     = INPUT_POSTS or "data/raw_posts.json"
        NO_MUSHROOM_IDS = NO_MUSHROOM_IDS or "data/no_mushroom_ids.json"
        OUTPUT_CSV      = OUTPUT_CSV or "data/photo_species.csv"
        CHECKPOINT      = CHECKPOINT or "data/photo_species_checkpoint.json"

    if app_config:
        API_URL = app_config.lm_studio_url
        MODEL   = app_config.lm_studio_model
    else:
        API_URL = API_URL or "http://localhost:1234/v1/chat/completions"
        MODEL   = MODEL or "google/gemma-3-12b"

    # Загружаем посты
    with open(INPUT_POSTS, encoding="utf-8") as f:
        posts = json.load(f)

    # Загружаем ID постов где модель не нашла грибов (из прошлых прогонов)
    no_mushroom_ids = set()
    if os.path.exists(NO_MUSHROOM_IDS):
        with open(NO_MUSHROOM_IDS, encoding="utf-8") as f:
            no_mushroom_ids = set(json.load(f))
        print(f"Постов без грибов (из прошлых прогонов): {len(no_mushroom_ids)}")

    # Фильтр: зима, текст-скипы
    from datetime import datetime
    WINTER_MONTHS = {11, 12, 1, 2, 3}

    to_process = []
    skipped_winter = 0
    skipped_text = 0
    skipped_no_mushroom = 0
    for p in posts:
        pid = str(p["id"])
        urls = p.get("photo_urls", [])
        if not urls:
            continue
        # Пропускаем посты где модель уже не нашла грибов
        if pid in no_mushroom_ids:
            skipped_no_mushroom += 1
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
            to_process.append({
                "id": pid,
                "urls": [urls[0], urls[-1]],
                "month": post_month,
            })
        else:
            to_process.append({
                "id": pid,
                "urls": [urls[0]],
                "month": post_month,
            })

    print(f"Модель: {MODEL}")
    print(f"API: {API_URL}")
    print(f"Постов с фото (апр-окт, релевантные): {len(to_process)}")
    print(f"Пропущено зимних: {skipped_winter}")
    print(f"Пропущено по тексту (рецепты, ягоды...): {skipped_text}")
    print(f"Пропущено (уже известно что без грибов): {skipped_no_mushroom}")

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

    # Проверяем LM Studio
    try:
        base_url = API_URL.rsplit("/chat/completions", 1)[0]
        requests.get(f"{base_url}/models", timeout=5)
    except Exception:
        print("LM Studio не запущена! Запусти сервер в LM Studio.")
        return

    # Параллельная обработка
    from concurrent.futures import ThreadPoolExecutor, as_completed
    N_WORKERS = 4

    import threading
    last_result_lock = threading.Lock()
    last_result = {"url": "", "answer": "", "time": 0}

    def process_one(post):
        all_results = {}  # species → sum count

        for url in post["urls"]:
            img = download_photo(url)
            if img is None:
                continue
            result = ask_model(img)
            if result is None:
                continue
            # Сохраняем для отображения
            with last_result_lock:
                last_result["url"] = url
                last_result["answer"] = str(result)[:150]
            for item in result:
                raw_sp = item.get("species", "")  # сырой ответ модели
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
                cnt = min(cnt, 150)  # клип
                # Сохраняем сырой вид — нормализация будет на этапе агрегации
                all_results[raw_sp] = max(all_results.get(raw_sp, 0), cnt)

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

                # Каждые 20 фото — показываем последний результат
                if pbar.n % 20 == 0:
                    with last_result_lock:
                        tqdm.write(f"  >> {last_result['answer']}")
                        tqdm.write(f"     {last_result['url']}")

            batch_start += BATCH

            # Чекпоинт после каждого батча
            save_checkpoint(results)
            recognized = sum(
                1 for r in results.values()
                if isinstance(r, list) and any(
                    item.get("species") not in ("ошибка", "none", "не_фото_грибов", None)
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
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

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
