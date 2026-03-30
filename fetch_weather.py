"""
Загрузка исторических данных о погоде для Санкт-Петербурга.
Источник: Open-Meteo Archive API (бесплатно, без ключа).

Координаты: 59.9343°N, 30.3351°E (центр СПб)
Вывод: data/weather_raw.csv     — сырые дневные показатели
       data/weather_features.csv — с лагами и производными фичами
"""

import os
import requests
import pandas as pd
from datetime import date, timedelta

LAT = 59.9343
LON = 30.3351
OUTPUT_RAW = "data/weather_raw.csv"
OUTPUT_FEATURES = "data/weather_features.csv"

# Запрашиваем данные с запасом (чуть раньше чем начало наших постов)
START_DATE = "2018-01-01"
END_DATE = date.today().isoformat()


# ── Загрузка сырых данных ──────────────────────────────────────────────────────

DAILY_VARS = [
    # Температура
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_max",        # ощущаемая
    "apparent_temperature_min",

    # Осадки
    "precipitation_sum",               # все осадки, мм
    "rain_sum",                        # дождь отдельно
    "snowfall_sum",                    # снег, см
    "precipitation_hours",             # часов осадков за день

    # Солнце и радиация
    "sunshine_duration",               # секунд солнца за день
    "shortwave_radiation_sum",         # МДж/м² — суммарная солнечная радиация

    # Ветер
    "wind_speed_10m_max",              # км/ч
    "wind_gusts_10m_max",
    "wind_direction_10m_dominant",     # градусы

    # Испарение
    "et0_fao_evapotranspiration",      # мм — потенциальное испарение
]

# Почасовые переменные — усредним до дневных
HOURLY_VARS = [
    "relative_humidity_2m",            # влажность воздуха, %
    "pressure_msl",                    # давление на уровне моря, гПа
    "soil_temperature_0cm",            # температура почвы у поверхности
    "soil_temperature_6cm",            # на глубине ~6 см
    "soil_moisture_0_to_1cm",          # влажность почвы (объёмная доля)
    "soil_moisture_1_to_3cm",
    "snow_depth",                      # высота снежного покрова, м
]


def fetch_daily(start: str, end: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Europe/Moscow",
    }
    print("Загружаем дневные данные (Open-Meteo)...")
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    data = r.json()["daily"]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    # Солнце: секунды → часы
    df["sunshine_hours"] = df["sunshine_duration"] / 3600
    df = df.drop(columns=["sunshine_duration"])
    return df


def fetch_hourly(start: str, end: str) -> pd.DataFrame:
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": LAT,
        "longitude": LON,
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Europe/Moscow",
    }
    print("Загружаем почасовые данные (Open-Meteo)...")
    r = requests.get(url, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()["hourly"]
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["time"]).dt.date
    df["date"] = pd.to_datetime(df["date"])
    # Усредняем по дням
    df_daily = df.drop(columns=["time"]).groupby("date").mean().reset_index()
    return df_daily


# ── Производные фичи ──────────────────────────────────────────────────────────

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)

    # Перепад температуры
    df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]

    # Флаг заморозка
    df["frost"] = (df["temperature_2m_min"] < 0).astype(int)

    # Накопленные осадки за N дней
    for window in [3, 7, 14, 30]:
        df[f"precip_{window}d"] = (
            df["precipitation_sum"].rolling(window, min_periods=1).sum()
        )

    # Скользящая средняя температуры
    for window in [3, 7, 14]:
        df[f"temp_mean_{window}d"] = (
            df["temperature_2m_mean"].rolling(window, min_periods=1).mean()
        )

    # Скользящая средняя влажности
    for window in [3, 7]:
        df[f"humidity_mean_{window}d"] = (
            df["relative_humidity_2m"].rolling(window, min_periods=1).mean()
        )

    # Лаговые значения температуры и осадков (T-1 ... T-7)
    for lag in [1, 2, 3, 5, 7, 10, 14]:
        df[f"temp_lag{lag}"] = df["temperature_2m_mean"].shift(lag)
        df[f"precip_lag{lag}"] = df["precipitation_sum"].shift(lag)

    # Дней без заморозка подряд (важно для грибного сезона)
    no_frost = (df["temperature_2m_min"] >= 0).astype(int)
    streak = []
    count = 0
    for v in no_frost:
        count = count + 1 if v else 0
        streak.append(count)
    df["days_no_frost"] = streak

    # Климатическая аномалия температуры: отклонение от среднего за этот день года
    df["day_of_year"] = df["date"].dt.day_of_year
    clim_mean = (
        df.groupby("day_of_year")["temperature_2m_mean"]
        .transform("mean")
    )
    df["temp_anomaly"] = df["temperature_2m_mean"] - clim_mean
    df = df.drop(columns=["day_of_year"])

    # Вспомогательные временные фичи
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.day_of_year
    df["weekday"] = df["date"].dt.weekday

    return df


# ── Основной pipeline ──────────────────────────────────────────────────────────

def main():
    os.makedirs("data", exist_ok=True)

    df_daily = fetch_daily(START_DATE, END_DATE)
    df_hourly = fetch_hourly(START_DATE, END_DATE)

    # Объединяем дневные и агрегированные почасовые
    df = df_daily.merge(df_hourly, on="date", how="left")

    df.to_csv(OUTPUT_RAW, index=False)
    print(f"Сырые данные сохранены: {OUTPUT_RAW} ({len(df)} дней, {len(df.columns)} колонок)")

    df_feat = add_features(df)
    df_feat.to_csv(OUTPUT_FEATURES, index=False)
    print(f"С фичами сохранено:     {OUTPUT_FEATURES} ({len(df_feat.columns)} колонок)")

    print(f"\nКолонки в итоговом датасете:")
    for col in df_feat.columns:
        print(f"  {col}")


if __name__ == "__main__":
    main()
