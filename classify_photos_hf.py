"""
Классификация грибов по фото через HuggingFace Transformers (Qwen2.5-VL).
Работает напрямую на GPU без Ollama.

Установка: pip install transformers torch torchvision qwen-vl-utils accelerate
Запуск:    python classify_photos_hf.py
"""

import os
import re
import json
import time
import base64
import requests
from io import BytesIO
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

INPUT_POSTS    = "data/raw_posts.json"
INPUT_SPECIES  = "data/posts_with_species.csv"
OUTPUT_CSV     = "data/photo_species.csv"
CHECKPOINT     = "data/photo_species_checkpoint.json"

WINTER_MONTHS = {11, 12, 1, 2, 3}

PROMPT = """Mushroom species and count? JSON only: [{"species":"name","count":N}]
No mushrooms: [{"species":"none","count":0}]
Basket=30-50, handful=5-10.
Common species: Boletus edulis, Cantharellus cibarius, Cantharellus tubaeformis, Armillaria, Suillus, Leccinum, Morchella, Gyromitra, Pleurotus, Russula, Lactarius"""

# Посты с этими словами пропускаем
SKIP_PHOTO_RE = re.compile("|".join([
    r"рецепт", r"приготовл", r"жарен|варен|тушен|маринов",
    r"суп\b|пирог|жульен|соус", r"продаж|куплю|продам|цена",
    r"отравлен|ядовит|опасн", r"мухомор", r"реклам|подписк|розыгрыш",
    r"карт[аыу]\s+грибн", r"прогноз\s+погод", r"стих|поэзи|цитат",
    r"конкурс|голосован", r"правил[аоы]\s+(груп|сообщ)",
    r"клещ|змея|змей|медвед",
    r"ягод[аыу]|клюкв|брусник|черник|морошк",
    r"рыбалк|рыб[аыу]|щук[аиу]|окун[ьяей]|удочк|спиннинг|улов[а-я]*\s+рыб|карас[ьяей]|лещ",
    r"ничего\s+не\s+наш", r"не\s+наш[ёел][иа]?\s+(ни|гриб)",
    r"пустой?\s+корзин", r"без\s+гриб", r"ни\s+одного\s+гриб",
    r"\bпролёт\b", r"\bвпустую\b", r"\bбезрезультат",
    r"с\s+днём?\s+рождени", r"день\s+рождени", r"\bдр\b", r"поздравля",
    r"с\s+новым\s+год", r"новогодн", r"8\s*марта", r"23\s*февраля", r"день\s+защитника",
]), re.IGNORECASE)

SPECIES_MAP = {
    "boletus edulis": "белый", "boletus": "белый",
    "cantharellus cibarius": "лисичка", "cantharellus": "лисичка",
    "cantharellus tubaeformis": "лисичка", "craterellus tubaeformis": "лисичка",
    "tubular chanterelle": "лисичка", "winter chanterelle": "лисичка",
    "yellowfoot": "лисичка", "craterellus": "лисичка",
    "armillaria": "опёнок", "armillaria mellea": "опёнок", "honey fungus": "опёнок",
    "suillus": "маслёнок", "suillus luteus": "маслёнок", "suillus granulatus": "маслёнок",
    "leccinum aurantiacum": "подосиновик", "leccinum": "подосиновик",
    "leccinum scabrum": "подберёзовик", "leccinum versipelle": "подосиновик",
    "leccinum vulpinum": "подосиновик", "leccinum albostipitatum": "подосиновик",
    "orange birch bolete": "подосиновик", "birch bolete": "подберёзовик",
    "red-capped scaber stalk": "подосиновик", "orange cap boletus": "подосиновик",
    "morchella": "сморчок", "morchella esculenta": "сморчок", "morel": "сморчок",
    "gyromitra": "строчок", "gyromitra esculenta": "строчок",
    "verpa": "сморчковая_шапочка", "verpa bohemica": "сморчковая_шапочка",
    "lactarius deliciosus": "рыжик", "lactarius": "рыжик",
    "lactifluus": "груздь", "lactarius resimus": "груздь",
    "xerocomus": "моховик", "xerocomellus": "моховик",
    "russula": "сыроежка",
    "pholiota": "опёнок", "kuehneromyces": "опёнок",
    "pleurotus": "вешенка", "pleurotus ostreatus": "вешенка", "oyster mushroom": "вешенка",
    "chanterelle": "лисичка", "porcini": "белый", "cep": "белый",
    "penny bun": "белый", "king bolete": "белый",
    "белый": "белый", "белый гриб": "белый", "боровик": "белый",
    "лисичка": "лисичка", "лисички": "лисичка",
    "опёнок": "опёнок", "опята": "опёнок", "опенок": "опёнок",
    "маслёнок": "маслёнок", "маслята": "маслёнок", "масленок": "маслёнок",
    "подосиновик": "подосиновик", "подберёзовик": "подберёзовик", "подберезовик": "подберёзовик",
    "сморчок": "сморчок", "сморчки": "сморчок",
    "строчок": "строчок", "строчки": "строчок",
    "сморчковая шапочка": "сморчковая_шапочка", "шапочка": "сморчковая_шапочка",
    "рыжик": "рыжик", "груздь": "груздь", "моховик": "моховик",
    "сыроежка": "сыроежка", "козляк": "козляк",
    "волнушка": "волнушка", "свинушка": "свинушка", "горькушка": "горькушка",
    "вешенка": "вешенка", "вёшенка": "вешенка",
    "лисичка_трубчатая": "лисичка",
    "none": "нет_грибов", "unknown": "другой", "другой": "другой",
    "нет_грибов": "нет_грибов", "нет грибов": "нет_грибов",
}


def normalize_species(name):
    return SPECIES_MAP.get(name.lower().strip(), "другой")


def load_model():
    """Загружает модель Qwen2.5-VL на GPU."""
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    import torch

    print("Загружаем модель Qwen2.5-VL-7B...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct",
        torch_dtype=torch.float16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
    print("Модель загружена!")
    return model, processor


def ask_model(model, processor, image_bytes):
    """Отправляет фото в модель, возвращает список видов."""
    import torch
    from PIL import Image

    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
        # Ресайз для скорости
        max_size = 720
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            image = image.resize((int(image.size[0] * ratio), int(image.size[1] * ratio)))

        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": PROMPT},
            ]}
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=False,
            )

        # Декодируем только новые токены
        generated = output_ids[0][inputs.input_ids.shape[1]:]
        response = processor.decode(generated, skip_special_tokens=True).strip()

        # Парсим JSON
        response = re.sub(r'"count"\s*:\s*(\d+)\s*-\s*(\d+)', r'"count": "\1-\2"', response)
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass
        json_match = re.search(r"\{[^}]+\}", response)
        if json_match:
            try:
                return [json.loads(json_match.group())]
            except json.JSONDecodeError:
                pass
        return None

    except Exception as e:
        print(f"  Model error: {e}")
        return None


def download_photo(url, timeout=15):
    try:
        resp = requests.get(url, timeout=timeout)
        if resp.status_code == 200 and len(resp.content) > 1000:
            return resp.content
        return None
    except Exception:
        return None


def load_checkpoint():
    if os.path.exists(CHECKPOINT):
        with open(CHECKPOINT, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(results):
    with open(CHECKPOINT, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)


def main():
    import csv

    with open(INPUT_POSTS, encoding="utf-8") as f:
        posts = json.load(f)

    # Фильтруем посты
    to_process = []
    skipped_winter = 0
    skipped_text = 0
    for p in posts:
        pid = str(p["id"])
        urls = p.get("photo_urls", [])
        if not urls:
            continue
        post_month = datetime.strptime(p["date_posted"], "%Y-%m-%d").month
        if post_month in WINTER_MONTHS:
            skipped_winter += 1
            continue
        text = p.get("text", "")
        if text and SKIP_PHOTO_RE.search(text):
            skipped_text += 1
            continue
        if len(urls) >= 2:
            to_process.append({"id": pid, "urls": [urls[0], urls[-1]], "month": post_month})
        else:
            to_process.append({"id": pid, "urls": [urls[0]], "month": post_month})

    print(f"Постов с фото (апр-окт, релевантные): {len(to_process)}")
    print(f"Пропущено зимних: {skipped_winter}")
    print(f"Пропущено по тексту: {skipped_text}")

    # Чекпоинт
    results = load_checkpoint()
    print(f"Уже обработано (чекпоинт): {len(results)}")
    remaining = [p for p in to_process if p["id"] not in results]
    print(f"Осталось обработать: {len(remaining)}")

    if not remaining:
        print("Всё обработано!")
        return

    # Загружаем модель
    model, processor = load_model()

    # Обрабатываем
    errors = 0
    pbar = tqdm(total=len(remaining), desc="Распознаём фото")

    for i, post in enumerate(remaining):
        all_results = {}

        for url in post["urls"]:
            img = download_photo(url)
            if img is None:
                continue
            result = ask_model(model, processor, img)
            if result is None:
                continue

            for item in result:
                sp = normalize_species(item.get("species", ""))
                cnt = item.get("count", 0)
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
                cnt = min(cnt, 150)
                all_results[sp] = max(all_results.get(sp, 0), cnt)

        if not all_results:
            results[post["id"]] = [{"species": "ошибка", "count": 0}]
            errors += 1
        else:
            results[post["id"]] = [{"species": sp, "count": cnt} for sp, cnt in all_results.items()]

        pbar.update(1)

        # Чекпоинт каждые 100
        if (i + 1) % 100 == 0:
            save_checkpoint(results)
            recognized = sum(
                1 for r in results.values()
                if isinstance(r, list) and any(
                    item.get("species") not in ("ошибка", "нет_грибов", None)
                    for item in r
                )
            )
            pbar.set_postfix(recognized=recognized, errors=errors)

    pbar.close()
    save_checkpoint(results)

    # Сохраняем CSV
    rows = []
    species_count = {}
    for pid, r in results.items():
        if not isinstance(r, list):
            continue
        for item in r:
            species = item.get("species", "ошибка")
            count = item.get("count", 0)
            try:
                count = int(count)
            except (ValueError, TypeError):
                count = 0
            rows.append({"id": pid, "photo_species": species, "photo_count": count})
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
