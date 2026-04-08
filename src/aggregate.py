"""
Агрегация: считаем количество отчётов по дате похода.
Вход:  data/posts_with_dates.csv
Выход: data/daily_counts.csv  — одна строка на дату, колонка report_count
"""

import pandas as pd
import os

INPUT_CSV = None
OUTPUT_CSV = None


def aggregate(city_config=None, app_config=None):
    global INPUT_CSV, OUTPUT_CSV

    if city_config:
        INPUT_CSV  = city_config.path("posts_with_dates.csv")
        OUTPUT_CSV = city_config.path("daily_counts.csv")
    else:
        INPUT_CSV  = INPUT_CSV or "data/posts_with_dates.csv"
        OUTPUT_CSV = OUTPUT_CSV or "data/daily_counts.csv"

    df = pd.read_csv(INPUT_CSV, parse_dates=["date_posted", "foray_date"])

    total = len(df)
    has_date = df["foray_date"].notna().sum()
    print(f"Постов всего: {total}, с датой похода: {has_date} ({has_date/total*100:.1f}%)")

    # Берём только посты с распознанной датой
    df_valid = df[df["foray_date"].notna()].copy()

    # Фильтр: исключаем явно неправдоподобные даты
    df_valid = df_valid[
        (df_valid["foray_date"].dt.year >= 2015) &
        (df_valid["foray_date"] <= pd.Timestamp.now())
    ]

    # Считаем отчёты по дате похода
    daily = (
        df_valid.groupby("foray_date")
        .agg(
            report_count=("id", "count"),
            avg_likes=("likes", "mean"),
            avg_views=("views", "mean"),
            median_views=("views", "median"),
            total_photos=("photos", "sum"),
        )
        .reset_index()
        .rename(columns={"foray_date": "date"})
        .sort_values("date")
    )

    # Сглаживание 20-х чисел (конкурсы/дайджесты) — заменяем на среднее соседей
    is_20th = (daily["date"].dt.day == 20) & (daily["date"].dt.month.isin([5, 6, 7, 8, 9]))
    n_20th = is_20th.sum()
    if n_20th > 0:
        prev = daily["report_count"].shift(1)
        nxt  = daily["report_count"].shift(-1)
        neighbor_avg = ((prev + nxt) / 2).fillna(prev).fillna(nxt).fillna(0)
        daily.loc[is_20th, "report_count"] = neighbor_avg.loc[is_20th].round().astype(int)
        print(f"\nЗаменено 20-х чисел (май-сен): {n_20th}")

    # Дополнительно: общие спайки > 3× медианы окна 7 дней
    rolling_median = daily["report_count"].rolling(7, center=True, min_periods=3).median()
    is_spike = daily["report_count"] > rolling_median * 3
    n_smoothed = is_spike.sum()
    if n_smoothed > 0:
        daily.loc[is_spike, "report_count"] = rolling_median[is_spike].astype(int)
        print(f"Сглажено прочих спайков: {n_smoothed}")

    # Скользящая медиана просмотров (30 дней) — прокси размера аудитории
    daily["audience_proxy"] = (
        daily["median_views"]
        .rolling(30, center=True, min_periods=7)
        .median()
        .round(0)
    )
    daily["audience_proxy"] = daily["audience_proxy"].bfill().ffill()

    # Нормализованный таргет: убираем эффект роста аудитории
    # baseline = медиана просмотров за 2020 год
    baseline_mask = daily["date"].dt.year == 2020
    if baseline_mask.any():
        baseline = daily.loc[baseline_mask, "audience_proxy"].median()
    else:
        baseline = daily["audience_proxy"].median()
    daily["audience_scale"] = (daily["audience_proxy"] / baseline).clip(lower=0.5)
    daily["mushroom_index"] = (daily["report_count"] / daily["audience_scale"]).round(2)

    # Убираем недельную сезонность (выходные vs будни)
    # Коэффициент дня недели: среднее по каждому weekday / общее среднее
    daily["_weekday"] = daily["date"].dt.weekday
    season_mask = daily["date"].dt.month.between(4, 11) & (daily["mushroom_index"] > 0)
    weekday_mean = daily.loc[season_mask].groupby("_weekday")["mushroom_index"].mean()
    overall_mean = daily.loc[season_mask, "mushroom_index"].mean()
    weekday_coeff = (weekday_mean / overall_mean).to_dict()
    daily["weekday_scale"] = daily["_weekday"].map(weekday_coeff).fillna(1.0)
    daily["mushroom_index_sm"] = (
        (daily["mushroom_index"] / daily["weekday_scale"])
        .rolling(3, center=True, min_periods=1)
        .mean()
        .round(2)
    )
    daily.drop(columns=["_weekday"], inplace=True)

    # Зима (ноябрь–март) = 0 — грибов нет, посты это шум
    WINTER_MONTHS = [11, 12, 1, 2, 3]
    is_winter = daily["date"].dt.month.isin(WINTER_MONTHS)
    daily.loc[is_winter, "mushroom_index"] = 0
    daily.loc[is_winter, "mushroom_index_sm"] = 0
    print(f"\nОбнулено зимних дней (ноя–мар): {is_winter.sum()}")

    print(f"\nWeekday coefficients:")
    for d, name in enumerate(["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]):
        print(f"  {name}: {weekday_coeff.get(d, 1.0):.2f}")
    print(f"\nAudience baseline ({2020}): {baseline:.0f} views")
    print(f"Audience scale: min={daily['audience_scale'].min():.2f}, max={daily['audience_scale'].max():.2f}")

    # Добавляем вспомогательные колонки
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["day_of_year"] = daily["date"].dt.day_of_year
    daily["weekday"] = daily["date"].dt.weekday   # 0=пн, 6=вс

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    daily.to_csv(OUTPUT_CSV, index=False)

    print(f"\nПервые 10 дней с наибольшим количеством отчётов:")
    print(daily.nlargest(10, "report_count")[["date", "report_count", "year"]].to_string(index=False))
    print(f"\nСохранено: {OUTPUT_CSV} ({len(daily)} уникальных дат)")

    # Краткая статистика по годам
    print("\nОтчёты по годам:")
    yearly = daily.groupby("year")["report_count"].sum().reset_index()
    print(yearly.to_string(index=False))


def main(city_config=None, app_config=None):
    aggregate(city_config=city_config, app_config=app_config)


if __name__ == "__main__":
    main()
