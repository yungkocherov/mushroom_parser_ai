"""
Загрузка сырых погодных данных из Open-Meteo Archive API.
Сохраняет только сырые данные в data/weather_raw.csv.
Повторный запуск докачивает только отсутствующие даты.

Запуск: python fetch_weather.py
"""

import os
import time
import requests
import pandas as pd
from datetime import date, timedelta

LAT = 59.9343
LON = 30.3351
OUTPUT_RAW = "data/weather_raw.csv"
START_DATE = "2018-01-01"
END_DATE = (date.today() - timedelta(days=1)).isoformat()

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min",
    "precipitation_sum", "rain_sum", "snowfall_sum", "precipitation_hours",
    "sunshine_duration", "shortwave_radiation_sum",
    "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
    "et0_fao_evapotranspiration",
]

HOURLY_VARS = [
    "relative_humidity_2m", "pressure_msl",
    "soil_temperature_0cm", "soil_temperature_6cm",
    "soil_moisture_0_to_1cm", "soil_moisture_1_to_3cm",
    "snow_depth",
]


def _get_with_retry(url: str, params: dict, retries: int = 5) -> requests.Response:
    for attempt in range(retries):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 429:
            wait = 10 * (attempt + 1)
            print(f"Rate limit, ждём {wait}с...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    raise RuntimeError("Превышен лимит попыток Open-Meteo")


def fetch_daily(start: str, end: str) -> pd.DataFrame:
    r = _get_with_retry("https://archive-api.open-meteo.com/v1/archive", {
        "latitude": LAT, "longitude": LON,
        "start_date": start, "end_date": end,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Europe/Moscow",
    })
    df = pd.DataFrame(r.json()["daily"])
    df["date"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    df["sunshine_hours"] = df["sunshine_duration"] / 3600
    df = df.drop(columns=["sunshine_duration"])
    return df


def fetch_hourly(start: str, end: str) -> pd.DataFrame:
    time.sleep(5)
    r = _get_with_retry("https://archive-api.open-meteo.com/v1/archive", {
        "latitude": LAT, "longitude": LON,
        "start_date": start, "end_date": end,
        "hourly": ",".join(HOURLY_VARS),
        "timezone": "Europe/Moscow",
    })
    df = pd.DataFrame(r.json()["hourly"])
    df["date"] = pd.to_datetime(df["time"]).dt.normalize()
    return df.drop(columns=["time"]).groupby("date").mean().reset_index()


def main():
    os.makedirs("data", exist_ok=True)

    # Определяем какие даты уже есть
    if os.path.exists(OUTPUT_RAW):
        existing = pd.read_csv(OUTPUT_RAW, parse_dates=["date"])
        last_date = existing["date"].max().date()
        fetch_start = (last_date + timedelta(days=1)).isoformat()

        if fetch_start > END_DATE:
            print(f"Данные актуальны (последняя дата: {last_date}). Ничего не качаем.")
            return

        print(f"Докачиваем с {fetch_start} по {END_DATE}...")
        df_daily  = fetch_daily(fetch_start, END_DATE)
        df_hourly = fetch_hourly(fetch_start, END_DATE)
        new_data  = df_daily.merge(df_hourly, on="date", how="left")
        df = pd.concat([existing, new_data], ignore_index=True)
    else:
        print(f"Качаем с {START_DATE} по {END_DATE}...")
        df_daily  = fetch_daily(START_DATE, END_DATE)
        df_hourly = fetch_hourly(START_DATE, END_DATE)
        df = df_daily.merge(df_hourly, on="date", how="left")

    df.to_csv(OUTPUT_RAW, index=False)
    print(f"Сохранено: {OUTPUT_RAW} ({len(df)} дней, {len(df.columns)} колонок)")


if __name__ == "__main__":
    main()
