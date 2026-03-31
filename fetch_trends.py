"""
Загрузка Google Trends для запроса "вк" (Россия).
Данные приходят еженедельно — интерполируем до дневного ряда.
Сохраняет data/google_trends.csv.

Запуск: python fetch_trends.py
"""

import time
import pandas as pd
from datetime import date
from pytrends.request import TrendReq

KEYWORD    = "вк"
GEO        = "RU"
START_DATE = "2020-01-01"
OUTPUT_CSV = "data/google_trends.csv"


def fetch_trends(keyword: str, start: str, end: str, geo: str = "RU") -> pd.DataFrame:
    """
    Скачивает Google Trends за весь период одним запросом (возвращает недельные данные).
    Нормализован к 0-100 внутри периода самим Google.
    """
    pytrends = TrendReq(hl="ru-RU", tz=180, timeout=(10, 30), retries=3, backoff_factor=1)
    timeframe = f"{start} {end}"
    print(f"  Запрос: '{keyword}' [{timeframe}] geo={geo}")
    pytrends.build_payload([keyword], geo=geo, timeframe=timeframe)
    df = pytrends.interest_over_time()

    if df.empty:
        raise RuntimeError("Google Trends вернул пустой результат")

    df = df[[keyword]].reset_index()
    df.columns = ["date", "vk_trend"]
    df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    end = date.today().isoformat()
    print(f"Загрузка Google Trends: '{KEYWORD}' (RU), {START_DATE}–{end}")

    weekly = fetch_trends(KEYWORD, START_DATE, end, geo=GEO)
    print(f"Получено недельных точек: {len(weekly)}")
    print(f"Диапазон значений: {weekly['vk_trend'].min()}–{weekly['vk_trend'].max()}")

    # Расширяем до дневного ряда через линейную интерполяцию
    date_range = pd.date_range(START_DATE, end, freq="D")
    df_daily = pd.DataFrame({"date": date_range})
    df_daily = df_daily.merge(weekly, on="date", how="left")
    df_daily["vk_trend"] = df_daily["vk_trend"].interpolate(method="linear").round(1)

    df_daily.to_csv(OUTPUT_CSV, index=False)
    print(f"Сохранено: {OUTPUT_CSV} ({len(df_daily)} дней)")


if __name__ == "__main__":
    main()
