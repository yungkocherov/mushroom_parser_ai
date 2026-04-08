"""
Панельная агрегация: (дата × группа грибов) → количество.

Вход:
  - data/posts_with_dates.csv  — посты с датами походов
  - data/photo_species.csv     — сырые ответы модели (от classify_photos.py)
  - data/weather_features.csv  — погодные фичи

Выход:
  - data/panel.csv — панельный датасет: одна строка = один день × одна группа

Запуск: python aggregate_panel.py
"""

import os
import pandas as pd
import numpy as np

INPUT_POSTS    = None
INPUT_PHOTOS   = None
INPUT_WEATHER  = None
INPUT_DAILY    = None
OUTPUT_PANEL   = None

# Маппинг сырых ответов модели → 4 целевые группы
# Модель отвечает: chanterelle, bolete, morel, honey_fungus, other, none
SPECIES_TO_GROUP = {
    "chanterelle": "лисичковые",
    "bolete": "болетовые",
    "morel": "весенние",
    "honey_fungus": "опята",
    # "other" и "none" — не входят в панель
}

TARGET_GROUPS = ["болетовые", "лисичковые", "весенние", "опята"]

SKIP_SPECIES = {"none", "other", "ошибка", "basket", ""}

WINTER_MONTHS = {11, 12, 1, 2, 3}


def load_photo_counts() -> pd.DataFrame:
    """Загружает результаты фото-классификации, джойнит с датами."""
    photos = pd.read_csv(INPUT_PHOTOS)
    posts = pd.read_csv(INPUT_POSTS, parse_dates=["date_posted", "foray_date"])

    # Дата: foray_date если есть, иначе date_posted
    posts["date"] = posts["foray_date"].fillna(posts["date_posted"])
    date_map = posts.set_index("id")["date"].to_dict()

    photos["id"] = photos["id"].astype(int)
    photos["date"] = photos["id"].map(date_map)
    photos = photos.dropna(subset=["date"])
    photos["date"] = pd.to_datetime(photos["date"])

    # Убираем невалидные даты
    photos = photos[(photos["date"] >= "2018-01-01") & (photos["date"] <= "2026-12-31")]

    print(f"  Всего записей в photo_species.csv: {len(photos)}")

    # Убираем нерелевантные виды
    photos = photos[~photos["photo_species"].isin(SKIP_SPECIES)]

    # Маппим сырой ответ → группа
    photos["group"] = photos["photo_species"].map(SPECIES_TO_GROUP)

    unmapped = photos["group"].isna().sum()
    if unmapped > 0:
        print(f"  Виды без группы (пропущены): {unmapped}")
        unknown = photos.loc[photos["group"].isna(), "photo_species"].value_counts().head(10)
        for sp, cnt in unknown.items():
            print(f"    {sp}: {cnt}")

    photos = photos.dropna(subset=["group"])
    print(f"  После маппинга в группы: {len(photos)}")

    # Сезонный фильтр:
    # - весенние только апрель-май
    # - лисичковые июнь-ноябрь (обычные + трубчатые объединены)
    # - болетовые, опята только июнь-октябрь
    month = photos["date"].dt.month
    spring_mask = (photos["group"] == "весенние") & (~month.isin([4, 5]))
    summer_mask = (photos["group"].isin(["болетовые", "опята"])) & (~month.isin(range(6, 11)))
    chanterelle_mask = (photos["group"] == "лисичковые") & (~month.isin(range(6, 12)))
    bad = spring_mask | summer_mask | chanterelle_mask
    n_filtered = bad.sum()
    photos = photos[~bad]
    print(f"  Отфильтровано по сезону: {n_filtered}")

    return photos[["date", "group", "photo_count"]].rename(columns={"group": "species"})


def build_panel(photos: pd.DataFrame) -> pd.DataFrame:
    """Строит панель (дата × группа) с нормализованным количеством."""

    # Агрегируем: сумма грибов по (дата, группа)
    daily_species = (
        photos.groupby(["date", "species"])["photo_count"]
        .sum()
        .reset_index()
        .rename(columns={"photo_count": "mushroom_count"})
    )

    # Нормализация по аудитории (из daily_counts.csv)
    daily_counts = pd.read_csv(INPUT_DAILY, parse_dates=["date"])
    if "audience_scale" in daily_counts.columns:
        scale_map = daily_counts.set_index("date")["audience_scale"].to_dict()
        daily_species["audience_scale"] = daily_species["date"].map(scale_map).fillna(1.0).clip(lower=0.5)
        daily_species["mushroom_count"] = (
            daily_species["mushroom_count"] / daily_species["audience_scale"]
        ).round(1)
        daily_species.drop(columns=["audience_scale"], inplace=True)
        print("  Нормализовано по audience_scale")
    else:
        print("  audience_scale не найден — без нормализации")

    # Полная сетка: все даты × все группы
    date_range = pd.date_range(
        daily_species["date"].min(),
        daily_species["date"].max(),
        freq="D",
    )
    grid = pd.MultiIndex.from_product(
        [date_range, TARGET_GROUPS],
        names=["date", "species"],
    ).to_frame(index=False)

    # Мёрджим — пустые ячейки = 0
    panel = grid.merge(daily_species, on=["date", "species"], how="left")
    panel["mushroom_count"] = panel["mushroom_count"].fillna(0)

    # Обнуляем зиму
    is_winter = panel["date"].dt.month.isin(WINTER_MONTHS)
    panel.loc[is_winter, "mushroom_count"] = 0

    # Нормализация по дню недели (для каждой группы отдельно)
    panel["weekday"] = panel["date"].dt.weekday
    season_mask = ~panel["date"].dt.month.isin(WINTER_MONTHS) & (panel["mushroom_count"] > 0)
    for sp in TARGET_GROUPS:
        sp_mask = (panel["species"] == sp) & season_mask
        if sp_mask.sum() < 50:
            continue
        weekday_mean = panel.loc[sp_mask].groupby("weekday")["mushroom_count"].mean()
        overall_mean = panel.loc[sp_mask, "mushroom_count"].mean()
        if overall_mean > 0:
            weekday_coeff = (weekday_mean / overall_mean).to_dict()
            sp_all = panel["species"] == sp
            scale = panel.loc[sp_all, "weekday"].map(weekday_coeff).fillna(1.0).clip(lower=0.3)
            panel.loc[sp_all, "mushroom_count"] = (
                panel.loc[sp_all, "mushroom_count"] / scale
            ).round(1)
    panel.drop(columns=["weekday"], inplace=True)
    print("  Нормализовано по дню недели")

    # Фильтрация выбросов: > 3× медианы окна 7 дней → среднее соседей
    n_outliers = 0
    for sp in TARGET_GROUPS:
        sp_mask = panel["species"] == sp
        vals = panel.loc[sp_mask, "mushroom_count"]
        rolling_med = vals.rolling(7, center=True, min_periods=3).median()
        # Только ненулевые медианы, иначе любое значение = выброс
        is_outlier = (vals > rolling_med * 3) & (rolling_med > 0) & (vals > 0)
        if is_outlier.sum() > 0:
            prev = vals.shift(1)
            nxt = vals.shift(-1)
            neighbor_avg = ((prev + nxt) / 2).fillna(prev).fillna(nxt).fillna(0)
            panel.loc[sp_mask & is_outlier, "mushroom_count"] = neighbor_avg[is_outlier].round(1)
            n_outliers += is_outlier.sum()
    print(f"  Сглажено выбросов (>3x медианы): {n_outliers}")

    # Сглаживание 3 дня (для каждой группы отдельно)
    for sp in TARGET_GROUPS:
        sp_mask = panel["species"] == sp
        panel.loc[sp_mask, "mushroom_count"] = (
            panel.loc[sp_mask, "mushroom_count"]
            .rolling(5, center=True, min_periods=1)
            .mean()
            .round(1)
        )
    print("  Сглажено (окно 3 дня)")

    return panel


def add_weather(panel: pd.DataFrame) -> pd.DataFrame:
    """Присоединяет погодные фичи — одинаковые для всех видов в один день."""
    weather = pd.read_csv(INPUT_WEATHER, parse_dates=["date"])
    panel = panel.merge(weather, on="date", how="left")
    return panel


def main(city_config=None, app_config=None):
    global INPUT_POSTS, INPUT_PHOTOS, INPUT_WEATHER, INPUT_DAILY, OUTPUT_PANEL

    if city_config:
        INPUT_POSTS   = city_config.path("posts_with_dates.csv")
        INPUT_PHOTOS  = city_config.path("photo_species.csv")
        INPUT_WEATHER = city_config.path("weather_features.csv")
        INPUT_DAILY   = city_config.path("daily_counts.csv")
        OUTPUT_PANEL  = city_config.path("panel.csv")
    else:
        INPUT_POSTS   = INPUT_POSTS or "data/posts_with_dates.csv"
        INPUT_PHOTOS  = INPUT_PHOTOS or "data/photo_species.csv"
        INPUT_WEATHER = INPUT_WEATHER or "data/weather_features.csv"
        INPUT_DAILY   = INPUT_DAILY or "data/daily_counts.csv"
        OUTPUT_PANEL  = OUTPUT_PANEL or "data/panel.csv"

    if not os.path.exists(INPUT_PHOTOS):
        print(f"Файл {INPUT_PHOTOS} не найден. Сначала запусти: python classify_photos.py")
        return

    print("Загружаем фото-классификацию...")
    photos = load_photo_counts()
    print(f"  Уникальных дат: {photos['date'].nunique()}")
    print(f"  Групп: {photos['species'].nunique()}")

    print("\nСтроим панель...")
    panel = build_panel(photos)
    print(f"  Размер: {len(panel)} строк ({panel['date'].nunique()} дней x {panel['species'].nunique()} групп)")

    print("\nПрисоединяем погоду...")
    panel = add_weather(panel)

    # Статистика по группам (только сезон)
    season = panel[~panel["date"].dt.month.isin(WINTER_MONTHS)]
    stats = (
        season.groupby("species")["mushroom_count"]
        .agg(["sum", "mean", "max"])
        .round(1)
        .sort_values("sum", ascending=False)
    )
    print(f"\nСтатистика по группам (сезон):")
    print(stats.to_string())

    # Фильтруем с 2018 года
    panel = panel[panel["date"].dt.year >= 2018].reset_index(drop=True)

    os.makedirs(os.path.dirname(OUTPUT_PANEL), exist_ok=True)
    panel.to_csv(OUTPUT_PANEL, index=False)
    print(f"\nСохранено: {OUTPUT_PANEL} ({len(panel)} строк, {len(panel.columns)} колонок)")


if __name__ == "__main__":
    main()

