"""
Построение фич из сырых погодных данных.
Читает data/weather_raw.csv, пишет data/weather_features.csv.

Запуск: python build_features.py
"""

import os
import numpy as np
import pandas as pd

INPUT_RAW       = None
OUTPUT_FEATURES = None


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True).copy()

    T    = df["temperature_2m_mean"]
    Tmin = df["temperature_2m_min"]
    Tmax = df["temperature_2m_max"]
    P    = df["precipitation_sum"]
    H    = df["relative_humidity_2m"]
    Pres = df["pressure_msl"]
    Soil = df["soil_temperature_0cm"]
    SM   = df["soil_moisture_0_to_1cm"]

    # ── Базовые флаги ─────────────────────────────────────────────────────────
    df["temp_range"]   = Tmax - Tmin
    df["frost"]        = (Tmin < 0).astype(int)
    df["frost_light"]  = ((Tmin >= -3) & (Tmin < 0)).astype(int)
    df["frost_hard"]   = (Tmin < -3).astype(int)
    df["warm_day"]     = (T >= 10).astype(int)
    df["hot_day"]      = (T >= 20).astype(int)
    df["rain_day"]     = (P > 0.5).astype(int)
    df["heavy_rain"]   = (P > 10).astype(int)

    # Лаги бинарных индикаторов (грибница реагирует с задержкой)
    for lag in [1, 2, 3, 5, 7]:
        df[f"warm_day_lag{lag}"] = df["warm_day"].shift(lag)
        df[f"hot_day_lag{lag}"]  = df["hot_day"].shift(lag)
        df[f"rain_day_lag{lag}"] = df["rain_day"].shift(lag)
        df[f"heavy_rain_lag{lag}"] = df["heavy_rain"].shift(lag)

    # ── Лаги температуры (1–21 день) ─────────────────────────────────────────
    for lag in range(1, 22):
        df[f"temp_lag{lag}"] = T.shift(lag)

    # ── Лаги осадков (1–14 дней) ─────────────────────────────────────────────
    for lag in range(1, 15):
        df[f"precip_lag{lag}"] = P.shift(lag)

    # ── Лаги влажности (1–7 дней) ────────────────────────────────────────────
    for lag in [1, 2, 3, 5, 7]:
        df[f"humidity_lag{lag}"] = H.shift(lag)

    # ── Лаги почвенной температуры ───────────────────────────────────────────
    for lag in [1, 3, 5, 7]:
        df[f"soil_temp_lag{lag}"] = Soil.shift(lag)

    # ── Скользящие mean / min / max температуры ───────────────────────────────
    for w in [3, 5, 7, 10, 14, 21, 30]:
        df[f"temp_mean_{w}d"] = T.rolling(w, min_periods=1).mean()
        df[f"temp_min_{w}d"]  = Tmin.rolling(w, min_periods=1).min()
        df[f"temp_max_{w}d"]  = Tmax.rolling(w, min_periods=1).max()

    # ── Волатильность и тренд температуры ─────────────────────────────────────
    for w in [3, 7, 14]:
        df[f"temp_std_{w}d"] = T.rolling(w, min_periods=2).std()

    df["temp_trend_3d"]  = df["temp_mean_3d"]  - df["temp_mean_7d"]
    df["temp_trend_7d"]  = df["temp_mean_7d"]  - df["temp_mean_14d"]
    df["temp_trend_14d"] = df["temp_mean_14d"] - df["temp_mean_21d"]

    # ── Накопленные осадки ────────────────────────────────────────────────────
    for w in [3, 5, 7, 10, 14, 21, 30, 45, 60]:
        df[f"precip_{w}d"] = P.rolling(w, min_periods=1).sum()

    # ── Скользящие средние влажности ─────────────────────────────────────────
    for w in [3, 5, 7, 14]:
        df[f"humidity_mean_{w}d"] = H.rolling(w, min_periods=1).mean()
    df["humidity_std_7d"] = H.rolling(7, min_periods=2).std()

    # ── Скользящие средние почвы ──────────────────────────────────────────────
    for w in [3, 7, 14]:
        df[f"soil_temp_mean_{w}d"]     = Soil.rolling(w, min_periods=1).mean()
        df[f"soil_moisture_mean_{w}d"] = SM.rolling(w, min_periods=1).mean()

    # ── Пороговые счётчики за N дней ─────────────────────────────────────────
    for w in [7, 14, 30]:
        df[f"rain_days_{w}d"]       = df["rain_day"].rolling(w, min_periods=1).sum()
        df[f"warm_days_{w}d"]       = df["warm_day"].rolling(w, min_periods=1).sum()
        df[f"frost_days_{w}d"]      = df["frost"].rolling(w, min_periods=1).sum()
        df[f"heavy_rain_days_{w}d"] = df["heavy_rain"].rolling(w, min_periods=1).sum()

    # ── Градусо-дни роста GDD (base=5°C) ─────────────────────────────────────
    gdd = (T - 5).clip(lower=0)
    for w in [7, 14, 21, 30]:
        df[f"gdd_{w}d"] = gdd.rolling(w, min_periods=1).sum()

    # ── Серии подряд идущих дней ──────────────────────────────────────────────
    def streak(cond):
        result, count = [], 0
        for v in cond:
            count = count + 1 if v else 0
            result.append(count)
        return result

    df["days_no_frost"]  = streak(Tmin >= 0)
    df["days_warm"]      = streak(T >= 10)
    df["days_with_rain"] = streak(P > 0.5)
    df["days_dry"]       = streak(P <= 0.5)
    df["days_above_15"]  = streak(T >= 15)

    # ── Давление ──────────────────────────────────────────────────────────────
    df["pressure_delta_1d"] = Pres - Pres.shift(1)
    df["pressure_delta_3d"] = Pres - Pres.shift(3)
    df["pressure_mean_7d"]  = Pres.rolling(7, min_periods=1).mean()

    # ── Временные фичи (нужны до аномалий) ───────────────────────────────────
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["weekday"]     = df["date"].dt.weekday
    df["is_weekend"]  = (df["weekday"] >= 5).astype(int)
    df["day_of_year"] = df["date"].dt.day_of_year

    # ── Климатические аномалии ────────────────────────────────────────────────
    for feat, src in [("temp_anomaly", T), ("precip_anomaly", P), ("humidity_anomaly", H)]:
        clim = df.groupby("day_of_year")[src.name].transform("mean")
        df[feat] = src - clim

    df["temp_anomaly_7d"]  = df["temp_mean_7d"]  - df.groupby("day_of_year")["temp_mean_7d"].transform("mean")
    df["temp_anomaly_14d"] = df["temp_mean_14d"] - df.groupby("day_of_year")["temp_mean_14d"].transform("mean")

    # Аномалии осадков и влажности за разные окна
    for w in [7, 14, 30]:
        col = f"precip_{w}d"
        df[f"precip_anomaly_{w}d"] = df[col] - df.groupby("day_of_year")[col].transform("mean")
    for w in [7, 14]:
        col = f"humidity_mean_{w}d"
        df[f"humidity_anomaly_{w}d"] = df[col] - df.groupby("day_of_year")[col].transform("mean")

    # Лагированные и сглаженные аномалии температуры
    for lag in [3, 7, 14]:
        df[f"temp_anomaly_lag{lag}"]    = df["temp_anomaly"].shift(lag)
        df[f"temp_anomaly_7d_lag{lag}"] = df["temp_anomaly_7d"].shift(lag)
    for w in [3, 7]:
        df[f"temp_anomaly_smooth{w}d"] = df["temp_anomaly"].rolling(w, min_periods=1).mean()

    # ── Праздники и выходные ──────────────────────────────────────────────────
    RU_HOLIDAYS = {
        (1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7),
        (2, 23), (3, 8), (5, 1), (5, 9), (6, 12), (11, 4),
    }
    month_day = list(zip(df["date"].dt.month, df["date"].dt.day))
    df["is_holiday"]   = [1 if md in RU_HOLIDAYS else 0 for md in month_day]
    df["is_day_off"]   = ((df["is_weekend"] == 1) | (df["is_holiday"] == 1)).astype(int)
    df["long_weekend"] = df["is_day_off"].rolling(3, min_periods=1).sum().ge(2).astype(int)

    # ── «Дождь → тепло и солнце» ──────────────────────────────────────────────
    for rain_lag in [3, 5, 7, 10]:
        heavy_past = df["heavy_rain"].shift(rain_lag).fillna(0)
        df[f"rain{rain_lag}d_then_warm"] = (heavy_past * (T >= 12) * (P <= 1)).astype(int)

    good_rain_past = (
        df["precip_lag3"].fillna(0) +
        df["precip_lag5"].fillna(0) +
        df["precip_lag7"].fillna(0)
    ) > 5
    for w in [5, 7, 10]:
        df[f"good_rain_then_warm_{w}d"] = (good_rain_past & (T >= 12)).astype(int)

    # ── Лаги паттерна "дождь → тепло → грибы через N дней" ────────────────────
    for rain_lag in [3, 5, 7, 10]:
        base = df[f"rain{rain_lag}d_then_warm"]
        for delay in [1, 2, 3, 5, 7]:
            df[f"rain{rain_lag}d_then_warm_lag{delay}"] = base.shift(delay)

    for w in [5, 7, 10]:
        base = df[f"good_rain_then_warm_{w}d"]
        for delay in [1, 2, 3, 5, 7]:
            df[f"good_rain_then_warm_{w}d_lag{delay}"] = base.shift(delay)

    # ── Взаимодействия ────────────────────────────────────────────────────────
    df["temp_x_humidity"]  = T * H / 100
    df["temp_x_precip_7d"] = df["temp_mean_7d"] * df["precip_7d"]
    df["warm_and_wet_7d"]  = df["warm_days_7d"]  * df["rain_days_7d"]
    df["gdd14_x_precip14"] = df["gdd_14d"]        * df["precip_14d"]

    # ── Фурье-признаки сезонности ─────────────────────────────────────────────
    doy = df["day_of_year"]
    for k in range(1, 6):  # 5 гармоник — ловят и плавные и резкие сезонные паттерны
        df[f"sin_doy{k}"] = np.sin(2 * k * np.pi * doy / 365)
        df[f"cos_doy{k}"] = np.cos(2 * k * np.pi * doy / 365)

    # ── Дополнительные календарные фичи ──────────────────────────────────────
    df["day_of_month"]  = df["date"].dt.day
    df["week_of_year"]  = df["date"].dt.isocalendar().week.astype(int)

    # Расстояние до пиковых дат сезона (20 авг = doy 232, 20 сен = doy 263)
    df["days_to_aug20"] = (doy - 232).abs()
    df["days_to_sep20"] = (doy - 263).abs()
    df["days_to_peak"]  = df[["days_to_aug20", "days_to_sep20"]].min(axis=1)

    # Интеракция: 20-е число в пиковые месяцы
    dom = df["day_of_month"]
    for m, name in [(5, 'may'), (6, 'jun'), (7, 'jul'), (8, 'aug'), (9, 'sep')]:
        df[f"is_around_20th_{name}"] = ((df["month"] == m) & (dom >= 18) & (dom <= 22)).astype(int)
    df["is_peak_window"] = (
        df["is_around_20th_may"] | df["is_around_20th_jun"] | df["is_around_20th_jul"] |
        df["is_around_20th_aug"] | df["is_around_20th_sep"]
    ).astype(int)

    # Грибной сезон: апрель–ноябрь (мягкий индикатор через гауссиану)
    # Центр = doy 240 (конец августа), sigma = 60 дней
    df["season_gauss"] = np.exp(-0.5 * ((doy - 240) / 60) ** 2).round(4)

    return df


def main(city_config=None, app_config=None):
    global INPUT_RAW, OUTPUT_FEATURES

    if city_config:
        INPUT_RAW       = city_config.path("weather_raw.csv")
        OUTPUT_FEATURES = city_config.path("weather_features.csv")
    else:
        INPUT_RAW       = INPUT_RAW or "data/weather_raw.csv"
        OUTPUT_FEATURES = OUTPUT_FEATURES or "data/weather_features.csv"

    if not os.path.exists(INPUT_RAW):
        print(f"Файл {INPUT_RAW} не найден. Сначала запусти: python fetch_weather.py")
        return

    df_raw = pd.read_csv(INPUT_RAW, parse_dates=["date"])
    print(f"Сырые данные: {len(df_raw)} дней, {len(df_raw.columns)} колонок")

    df_feat = add_features(df_raw)

    os.makedirs(os.path.dirname(OUTPUT_FEATURES), exist_ok=True)
    df_feat.to_csv(OUTPUT_FEATURES, index=False)
    print(f"Фичи сохранены: {OUTPUT_FEATURES} ({len(df_feat.columns)} колонок, {len(df_feat)} дней)")


if __name__ == "__main__":
    main()
