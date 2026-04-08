"""
Сбор постов из группы ВК через VK API.
Сохраняет сырые данные в data/{city}/raw_posts.json
"""

import os
import json
import time
import csv
from datetime import datetime, timezone
from dotenv import load_dotenv
import requests
from tqdm import tqdm

load_dotenv()

VK_TOKEN = os.getenv("VK_TOKEN")
API_VERSION = "5.131"
BATCH_SIZE = 100
DELAY = 0.35
CHECKPOINT_EVERY = 500

# Эти переменные устанавливаются в main() из CityConfig
GROUP_DOMAIN = None
YEARS_BACK = None
OUTPUT_DIR = None
OUTPUT_JSON = None
OUTPUT_CSV = None
CHECKPOINT_FILE = None


def vk_request(method: str, params: dict, retries: int = 5) -> dict:
    """Выполняет запрос к VK API с повторными попытками при сетевых ошибках."""
    url = f"https://api.vk.com/method/{method}"
    params = {**params, "access_token": VK_TOKEN, "v": API_VERSION}
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if "error" in data:
                raise RuntimeError(f"VK API error {data['error']['error_code']}: {data['error']['error_msg']}")
            return data["response"]
        except (requests.exceptions.ConnectionError,
                requests.exceptions.Timeout) as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt  # 1, 2, 4, 8, 16 секунд
            print(f"\nСетевая ошибка (попытка {attempt+1}/{retries}), ждём {wait}с: {e}")
            time.sleep(wait)


def get_total_count() -> int:
    resp = vk_request("wall.get", {"domain": GROUP_DOMAIN, "count": 1, "filter": "owner"})
    return resp["count"]


def load_checkpoint() -> tuple[list[dict], int]:
    """Загружает сохранённый прогресс, если есть."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, encoding="utf-8") as f:
            data = json.load(f)
        posts = data["posts"]
        offset = data["offset"]
        print(f"Найден checkpoint: {len(posts)} постов, offset={offset}. Продолжаем...")
        return posts, offset
    return [], 0


def save_checkpoint(posts: list[dict], offset: int):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(CHECKPOINT_FILE, "w", encoding="utf-8") as f:
        json.dump({"posts": posts, "offset": offset}, f, ensure_ascii=False)


def fetch_all_posts(cutoff_ts: int) -> list[dict]:
    """
    Скачивает все посты группы, начиная с самых новых.
    Останавливается, когда дата поста уходит раньше cutoff_ts.
    Поддерживает продолжение после обрыва через checkpoint.
    """
    total = get_total_count()
    print(f"Всего постов в группе: {total}")

    posts, offset = load_checkpoint()
    stopped_early = False
    last_checkpoint_size = len(posts)

    try:
        with tqdm(total=total, initial=offset, desc="Скачиваем посты", unit="пост") as pbar:
            while True:
                resp = vk_request("wall.get", {
                    "domain": GROUP_DOMAIN,
                    "count": BATCH_SIZE,
                    "offset": offset,
                    "filter": "owner",
                })
                batch = resp.get("items", [])
                if not batch:
                    break

                for post in batch:
                    if post["date"] < cutoff_ts:
                        stopped_early = True
                        break
                    posts.append({
                        "id": post["id"],
                        "date_ts": post["date"],
                        "date_posted": datetime.fromtimestamp(
                            post["date"], tz=timezone.utc
                        ).strftime("%Y-%m-%d"),
                        "text": post.get("text", ""),
                        "likes": post.get("likes", {}).get("count", 0),
                        "reposts": post.get("reposts", {}).get("count", 0),
                        "views": post.get("views", {}).get("count", 0),
                        "photos": sum(
                            1 for a in post.get("attachments", [])
                            if a.get("type") == "photo"
                        ),
                        # URL фото в максимальном доступном разрешении
                        # Пригодятся для анализа видов грибов через CV/LLM
                        "photo_urls": [
                            max(
                                a["photo"]["sizes"],
                                key=lambda s: s["width"] * s["height"]
                            )["url"]
                            for a in post.get("attachments", [])
                            if a.get("type") == "photo"
                               and a.get("photo", {}).get("sizes")
                        ],
                    })

                pbar.update(len(batch))
                offset += len(batch)

                # Checkpoint каждые CHECKPOINT_EVERY постов
                if len(posts) - last_checkpoint_size >= CHECKPOINT_EVERY:
                    save_checkpoint(posts, offset)
                    last_checkpoint_size = len(posts)

                if stopped_early or offset >= total:
                    break

                time.sleep(DELAY)

    except Exception:
        # Сохраняем прогресс при любой ошибке
        if posts:
            print(f"\nОшибка — сохраняем checkpoint ({len(posts)} постов)...")
            save_checkpoint(posts, offset)
        raise

    # Удаляем checkpoint после успешного завершения
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)

    return posts


def save_posts(posts: list[dict]):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # JSON
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(posts, f, ensure_ascii=False, indent=2)
    print(f"JSON сохранён: {OUTPUT_JSON} ({len(posts)} постов)")

    # CSV
    if posts:
        with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=posts[0].keys())
            writer.writeheader()
            writer.writerows(posts)
        print(f"CSV сохранён: {OUTPUT_CSV}")


def main(city_config=None, app_config=None):
    global GROUP_DOMAIN, YEARS_BACK, OUTPUT_DIR, OUTPUT_JSON, OUTPUT_CSV, CHECKPOINT_FILE

    if city_config:
        GROUP_DOMAIN = city_config.vk_group
        YEARS_BACK = city_config.years_back
        OUTPUT_DIR = str(city_config.data_dir)
        OUTPUT_JSON = city_config.path("raw_posts.json")
        OUTPUT_CSV = city_config.path("raw_posts.csv")
        CHECKPOINT_FILE = city_config.path("checkpoint.json")
    else:
        # Fallback для автономного запуска
        GROUP_DOMAIN = GROUP_DOMAIN or "grib_spb"
        YEARS_BACK = YEARS_BACK or 8
        OUTPUT_DIR = OUTPUT_DIR or "data"
        OUTPUT_JSON = OUTPUT_JSON or os.path.join(OUTPUT_DIR, "raw_posts.json")
        OUTPUT_CSV = OUTPUT_CSV or os.path.join(OUTPUT_DIR, "raw_posts.csv")
        CHECKPOINT_FILE = CHECKPOINT_FILE or os.path.join(OUTPUT_DIR, "checkpoint.json")

    if not VK_TOKEN:
        raise EnvironmentError("VK_TOKEN не найден — проверь файл .env")

    cutoff_dt = datetime.now(tz=timezone.utc).replace(
        year=datetime.now().year - YEARS_BACK
    )
    cutoff_ts = int(cutoff_dt.timestamp())
    print(f"[{GROUP_DOMAIN}] Собираем посты с {cutoff_dt.strftime('%Y-%m-%d')} по сегодня")

    posts = fetch_all_posts(cutoff_ts)
    print(f"\nСобрано постов в диапазоне дат: {len(posts)}")

    save_posts(posts)


if __name__ == "__main__":
    main()
