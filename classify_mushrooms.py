"""
Классификация грибов по тексту поста.
Один пост может содержать несколько видов.

Запуск: python classify_mushrooms.py
"""

import re
import csv
import os
from collections import Counter

INPUT_CSV  = "data/posts_with_dates.csv"
OUTPUT_CSV = "data/posts_with_species.csv"

# ── Паттерны для каждого вида ────────────────────────────────────────────────
# Каждый паттерн компилируется с re.IGNORECASE
# Используем word boundaries и основы слов для покрытия падежей

SPECIES = {
    "белый": [
        r"\bбел[ыоа][йхем]\b",         # белый, белых, белые, белого, белому
        r"\bбелен[ьк]",                 # беленький
        r"\bборовик",                   # боровик, боровики, боровиков
        r"\bболет",                     # болетус (латинское)
        r"\bтолстоног",                 # толстоногий, толстоногие (сленг)
        r"\bцарь.{0,5}гриб",           # царь грибов
        r"\bбеляк",                     # беляк (сленг)
        r"\bбелёш",                     # белёша (ласк.)
    ],
    "лисичка": [
        r"\bлисич",                     # лисичка, лисички, лисичек
        r"\bлисёнок",                   # лисёнок (ласк.)
        r"\bпетуш[оки]",               # петушки (редко, для лисичек)
        r"\bcantharellus\b",            # латинское
    ],
    "опёнок": [
        r"\bопят",                      # опята, опят
        r"\bопён",                      # опёнок, опёнки
        r"\bопен",                      # опенок (без ё)
        r"\bармиллярия",                # латинское (Armillaria)
    ],
    "рыжик": [
        r"\bрыжик",                     # рыжик, рыжики, рыжиков
        r"\bрыженьк",                   # рыженький (ласк.)
        r"\bцарск[иой].{0,5}гриб",     # царский гриб (иногда рыжик)
    ],
    "маслёнок": [
        r"\bмаслят",                    # маслята, маслят
        r"\bмаслён",                    # маслёнок, маслёнки
        r"\bмаслен",                    # масленок (без ё)
        r"\bмаслик",                    # маслики (сленг)
    ],
    "подосиновик": [
        r"\bподосиновик",               # подосиновик, подосиновики
        r"\bкрасноголовик",             # красноголовик
        r"\bкрасн[еоы][нйхм]",         # красный, красные, красненький — КОНТЕКСТ!
        r"\bкрасненьк",                 # красненький, красненькие
        r"\bобабок",                    # обабок (может быть и подберёзовик)
        r"\bобабк",                     # обабки
        r"\bлекцинум",                  # латинское (Leccinum)
    ],
    "подберёзовик": [
        r"\bподберёзовик",              # подберёзовик
        r"\bподберезовик",              # без ё
        r"\bчерноголовик",              # черноголовик (сленг)
        r"\bберёзовик",                 # берёзовик
        r"\bберезовик",                 # без ё
    ],
    "сморчок": [
        r"\bсморч[оки]",               # сморчок, сморчки, сморчков
        r"\bmorchella\b",               # латинское
    ],
    "сморчковая_шапочка": [
        r"\bшапочк",                    # шапочка, шапочки (в контексте грибов)
        r"\bсморчков[аоы][йяе].{0,3}шапочк",  # сморчковая шапочка
        r"\bверп[аы]\b",               # верпа (Verpa)
    ],
    "груздь": [
        r"\bгрузд",                     # груздь, грузди, груздей
        r"\bгруздочк",                  # груздочки (ласк.)
    ],
    "волнушка": [
        r"\bволнушк",                   # волнушка, волнушки
        r"\bволнух",                    # волнухи (сленг)
    ],
    "сыроежка": [
        r"\bсыроежк",                   # сыроежка, сыроежки
        r"\bсыроег",                    # сыроеги (разг.)
    ],
    "моховик": [
        r"\bмоховик",                   # моховик, моховики
    ],
    "козляк": [
        r"\bкозляк",                    # козляк, козляки
        r"\bкозлят",                    # козлята (разг.)
    ],
    "горькушка": [
        r"\bгорькушк",                  # горькушка, горькушки
    ],
    "свинушка": [
        r"\bсвинушк",                   # свинушка, свинушки
        r"\bсвинух",                    # свинухи (разг.)
    ],
}

# "красный/красные" слишком общее — проверяем контекст
AMBIGUOUS_RED = re.compile(
    r"\bкрасн[еоы][нйхм]",
    re.IGNORECASE,
)
RED_CONTEXT = re.compile(
    r"красн\w{0,10}\s{0,3}(гриб|подосинов|шляпк|найти|собрал|нашёл|нашел|корзин|ведр|штук|лес)",
    re.IGNORECASE,
)

# Компилируем
COMPILED = {}
for species, patterns in SPECIES.items():
    COMPILED[species] = [re.compile(p, re.IGNORECASE) for p in patterns]


def classify_text(text: str) -> list[str]:
    """Возвращает список видов грибов, найденных в тексте."""
    if not text or not text.strip():
        return []

    found = []
    for species, patterns in COMPILED.items():
        # Специальная обработка "красных" для подосиновика
        if species == "подосиновик":
            for p in patterns:
                if p.pattern.startswith(r"\bкрасн"):
                    # "красный" считаем подосиновиком только в грибном контексте
                    if RED_CONTEXT.search(text):
                        found.append(species)
                        break
                elif p.search(text):
                    found.append(species)
                    break
        else:
            for p in patterns:
                if p.search(text):
                    found.append(species)
                    break

    return found


def main():
    with open(INPUT_CSV, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Постов: {len(rows)}")

    # Классифицируем
    species_counter = Counter()
    posts_with_species = 0
    multi_species = 0

    for row in rows:
        text = row.get("text_preview", "")
        species = classify_text(text)
        row["species"] = "|".join(species) if species else ""

        if species:
            posts_with_species += 1
            for s in species:
                species_counter[s] += 1
            if len(species) > 1:
                multi_species += 1

    # Сохраняем
    os.makedirs("data", exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Статистика
    has_date = sum(1 for r in rows if r.get("foray_date"))
    print(f"\nПостов с датой: {has_date}")
    print(f"Постов с определённым видом: {posts_with_species} ({posts_with_species/len(rows)*100:.1f}%)")
    print(f"Постов с несколькими видами: {multi_species}")
    print(f"\nВиды грибов:")
    for species, count in species_counter.most_common():
        print(f"  {species:25s} {count:6d}")

    # Посты с датой, но без вида — для анализа
    no_species = [r for r in rows if r.get("foray_date") and not r.get("species")]
    print(f"\nПостов с датой, но без определённого вида: {len(no_species)}")

    # Частотный анализ слов в нераспознанных (для поиска нового сленга)
    word_freq = Counter()
    for r in no_species[:5000]:
        text = r.get("text_preview", "").lower()
        words = re.findall(r"[а-яёА-ЯЁ]{4,}", text)
        word_freq.update(words)

    # Убираем стоп-слова
    stop = {"этот", "этого", "было", "были", "была", "есть", "очень", "сегодня",
            "вчера", "более", "можно", "после", "будет", "всего", "когда", "тоже",
            "грибы", "гриб", "грибов", "грибной", "грибочки", "поход", "лесу", "леса",
            "нашли", "нашёл", "нашел", "нашла", "собрали", "собрал", "ходили", "пошли",
            "сходил", "сходили", "место", "места", "район", "выборг", "всево", "приоз"}

    print(f"\nТоп-40 слов в постах без определённого вида (для поиска сленга):")
    for word, count in word_freq.most_common(80):
        if word not in stop and count >= 10:
            print(f"  {word:20s} {count}")


if __name__ == "__main__":
    main()
