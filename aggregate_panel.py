"""
Панельная агрегация: (дата × вид гриба) → количество.

Вход:
  - data/posts_with_dates.csv  — посты с датами
  - data/photo_species.csv     — виды грибов по фото (от classify_photos.py)
  - data/weather_features.csv  — погодные фичи

Выход:
  - data/panel.csv — панельный датасет: одна строка = один день × один вид

Запуск: python aggregate_panel.py
"""

import os
import pandas as pd
import numpy as np

INPUT_POSTS    = "data/posts_with_dates.csv"
INPUT_PHOTOS   = "data/photo_species.csv"
INPUT_WEATHER  = "data/weather_features.csv"
OUTPUT_PANEL   = "data/panel.csv"

# Виды, которые моделируем (остальные → "другой")
TARGET_SPECIES = [
    "белый", "лисичка", "опёнок", "маслёнок",
    "подосиновик", "подберёзовик", "сморчок",
    "сморчковая_шапочка", "рыжик", "груздь",
    "моховик",
]

SKIP_SPECIES = {"не_фото_грибов", "ошибка", "другой", ""}

# Зимние месяцы — обнуляем
WINTER_MONTHS = {11, 12, 1, 2, 3}


def load_photo_counts() -> pd.DataFrame:
    """Загружает результаты фото-классификации, джойнит с датами."""
    photos = pd.read_csv(INPUT_PHOTOS)
    posts = pd.read_csv(INPUT_POSTS, parse_dates=["date_posted", "foray_date"])

    # id → foray_date
    date_map = posts.set_index("id")["foray_date"].to_dict()

    photos["id"] = photos["id"].astype(int)
    photos["foray_date"] = photos["id"].map(date_map)
    photos = photos.dropna(subset=["foray_date"])
    photos["foray_date"] = pd.to_datetime(photos["foray_date"])

    # Убираем нерелевантные
    photos = photos[~photos["photo_species"].isin(SKIP_SPECIES)]

    # Нормализуем виды: редкие → "другой"
    photos.loc[~photos["photo_species"].isin(TARGET_SPECIES), "photo_species"] = "другой"

    return photos[["foray_date", "photo_species", "photo_count"]]


def build_panel(photos: pd.DataFrame) -> pd.DataFrame:
    """Строит панель (дата × вид) с суммарным количеством."""

    # Агрегируем: сумма грибов по (дата, вид)
    daily_species = (
        photos.groupby(["foray_date", "photo_species"])["photo_count"]
        .sum()
        .reset_index()
        .rename(columns={
            "foray_date": "date",
            "photo_species": "species",
            "photo_count": "mushroom_count",
        })
    )

    # Создаём полную сетку: все даты × все виды
    date_range = pd.date_range(
        daily_species["date"].min(),
        daily_species["date"].max(),
        freq="D",
    )
    all_species = TARGET_SPECIES + ["другой"]
    grid = pd.MultiIndex.from_product(
        [date_range, all_species],
        names=["date", "species"],
    ).to_frame(index=False)

    # Мёрджим — пустые ячейки = 0
    panel = grid.merge(daily_species, on=["date", "species"], how="left")
    panel["mushroom_count"] = panel["mushroom_count"].fillna(0).astype(int)

    # Обнуляем зиму
    is_winter = panel["date"].dt.month.isin(WINTER_MONTHS)
    panel.loc[is_winter, "mushroom_count"] = 0

    return panel


def add_weather(panel: pd.DataFrame) -> pd.DataFrame:
    """Присоединяет погодные фичи — одинаковые для всех видов в один день."""
    weather = pd.read_csv(INPUT_WEATHER, parse_dates=["date"])
    panel = panel.merge(weather, on="date", how="left")
    return panel


def main():
    if not os.path.exists(INPUT_PHOTOS):
        print(f"Файл {INPUT_PHOTOS} не найден. Сначала запусти: python classify_photos.py")
        return

    print("Загружаем фото-классификацию...")
    photos = load_photo_counts()
    print(f"  Записей (пост × вид): {len(photos)}")
    print(f"  Уникальных дат: {photos['foray_date'].nunique()}")
    print(f"  Виды: {photos['photo_species'].nunique()}")

    print("\nСтроим панель...")
    panel = build_panel(photos)
    print(f"  Размер панели: {len(panel)} строк ({panel['date'].nunique()} дней × {panel['species'].nunique()} видов)")

    print("\nПрисоединяем погоду...")
    panel = add_weather(panel)

    # Статистика по видам
    season = panel[~panel["date"].dt.month.isin(WINTER_MONTHS)]
    stats = (
        season.groupby("species")["mushroom_count"]
        .agg(["sum", "mean", "max"])
        .round(1)
        .sort_values("sum", ascending=False)
    )
    print(f"\nСтатистика по видам (сезон):")
    print(stats.to_string())

    # Фильтруем с 2020 года
    panel = panel[panel["date"].dt.year >= 2020].reset_index(drop=True)

    os.makedirs("data", exist_ok=True)
    panel.to_csv(OUTPUT_PANEL, index=False)
    print(f"\nСохранено: {OUTPUT_PANEL} ({len(panel)} строк, {len(panel.columns)} колонок)")


if __name__ == "__main__":
    main()
