"""
Сравнение qwen2.5vl:7b и qwen3-vl:8b на 100 случайных фото.
Результат сохраняется в data/model_comparison.csv

Запуск: python compare_models.py
"""

import os
import json
import re
import random
import base64
import requests
import csv
from datetime import datetime
from tqdm import tqdm

INPUT_POSTS = "data/raw_posts.json"
OUTPUT_CSV  = "data/model_comparison.csv"

WINTER_MONTHS = {11, 12, 1, 2, 3}
N_SAMPLES = 50

PROMPT = """Mushroom species and count? JSON only: [{"species":"name","count":N}]
No mushrooms: [{"species":"none","count":0}]
Basket=30-50, handful=5-10.
Common species: Boletus edulis, Cantharellus cibarius, Cantharellus tubaeformis, Armillaria, Suillus, Leccinum, Morchella, Gyromitra, Pleurotus, Russula, Lactarius"""

MODELS = {
    "qwen2.5vl:7b": {
        "endpoint": "/api/generate",
        "build_request": lambda img_b64: {
            "model": "qwen2.5vl:7b",
            "prompt": PROMPT,
            "images": [img_b64],
            "stream": False,
            "options": {"temperature": 0.1},
        },
        "parse_response": lambda resp: resp.json().get("response", "ERROR"),
    },
    "qwen3-vl:8b": {
        "endpoint": "/api/chat",
        "build_request": lambda img_b64: {
            "model": "qwen3-vl:8b",
            "messages": [{"role": "user", "content": PROMPT, "images": [img_b64]}],
            "stream": False,
            "options": {"temperature": 0.1},
        },
        "parse_response": lambda resp: resp.json().get("message", {}).get("content",
                                        resp.json().get("error", "ERROR")),
    },
}


def download_photo(url, timeout=15):
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200 and len(resp.content) > 1000:
            return resp.content
        return None
    except Exception:
        return None


def ask_model(model_name, img_b64):
    cfg = MODELS[model_name]
    url = f"http://localhost:11434{cfg['endpoint']}"
    try:
        resp = requests.post(url, json=cfg["build_request"](img_b64), timeout=50)
        if resp.status_code != 200:
            return f"ERROR: {resp.status_code}"
        return cfg["parse_response"](resp)
    except Exception as e:
        return f"ERROR: {e}"


def parse_json_response(text):
    """Извлекает JSON из ответа модели."""
    text = re.sub(r'"count"\s*:\s*(\d+)\s*-\s*(\d+)', r'"count": "\1-\2"', text)
    json_match = re.search(r"\[.*\]", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    json_match = re.search(r"\{[^}]+\}", text)
    if json_match:
        try:
            return [json.loads(json_match.group())]
        except json.JSONDecodeError:
            pass
    return text


def format_result(parsed):
    """Форматирует результат в читаемую строку."""
    if isinstance(parsed, list):
        parts = []
        for item in parsed:
            sp = item.get("species", "?")
            cnt = item.get("count", "?")
            parts.append(f"{sp}={cnt}")
        return ", ".join(parts)
    return str(parsed)


def main():
    with open(INPUT_POSTS, encoding="utf-8") as f:
        posts = json.load(f)

    # Фильтруем: только сезонные с фото
    seasonal = []
    for p in posts:
        urls = p.get("photo_urls", [])
        if not urls:
            continue
        month = datetime.strptime(p["date_posted"], "%Y-%m-%d").month
        if month in WINTER_MONTHS:
            continue
        seasonal.append(p)

    random.seed(42)
    sample = random.sample(seasonal, min(N_SAMPLES, len(seasonal)))
    print(f"Выбрано {len(sample)} фото для сравнения")

    # Подготовка: скачиваем все фото
    print("Скачиваем фото...")
    photos = []
    for p in tqdm(sample, desc="Скачиваем"):
        url = p["photo_urls"][-1]
        img = download_photo(url)
        if img is None:
            continue
        photos.append({
            "post_id": p["id"],
            "date": p["date_posted"],
            "photo_url": url,
            "img_b64": base64.b64encode(img).decode(),
        })
    print(f"Скачано: {len(photos)} фото")

    # Проход 1: qwen3-vl:8b
    print("\n=== Проход 1: qwen3-vl:8b ===")
    for i, photo in enumerate(tqdm(photos, desc="qwen3-vl:8b")):
        raw = ask_model("qwen3-vl:8b", photo["img_b64"])
        parsed = parse_json_response(raw)
        photo["qwen3-vl:8b_raw"] = raw[:200]
        photo["qwen3-vl:8b_result"] = format_result(parsed)
        if (i + 1) % 10 == 0:
            tqdm.write(f"  {photo['qwen3-vl:8b_result']}  |  {photo['photo_url'][:80]}...")

    # Проход 2: qwen2.5vl:7b
    print("\n=== Проход 2: qwen2.5vl:7b ===")
    for i, photo in enumerate(tqdm(photos, desc="qwen2.5vl:7b")):
        raw = ask_model("qwen2.5vl:7b", photo["img_b64"])
        parsed = parse_json_response(raw)
        photo["qwen2.5vl:7b_raw"] = raw[:200]
        photo["qwen2.5vl:7b_result"] = format_result(parsed)
        if (i + 1) % 10 == 0:
            tqdm.write(f"  {photo['qwen2.5vl:7b_result']}  |  {photo['photo_url'][:80]}...")

    results = photos

    # Сохраняем (без img_b64)
    os.makedirs("data", exist_ok=True)
    fieldnames = ["post_id", "date", "photo_url",
                  "qwen3-vl:8b_result", "qwen2.5vl:7b_result",
                  "qwen3-vl:8b_raw", "qwen2.5vl:7b_raw"]
    save_rows = [{k: v for k, v in r.items() if k != "img_b64"} for r in results]
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(save_rows)

    print(f"\nСохранено: {OUTPUT_CSV} ({len(results)} строк)")
    print("Открой CSV и сравни результаты с фото по URL")


if __name__ == "__main__":
    main()
