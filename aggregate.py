"""
Агрегация: считаем количество отчётов по дате похода.
Вход:  data/posts_with_dates.csv
Выход: data/daily_counts.csv  — одна строка на дату, колонка report_count
"""

import pandas as pd
import os

INPUT_CSV = "data/posts_with_dates.csv"
OUTPUT_CSV = "data/daily_counts.csv"


def aggregate():
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
            total_photos=("photos", "sum"),
        )
        .reset_index()
        .rename(columns={"foray_date": "date"})
        .sort_values("date")
    )

    # Добавляем вспомогательные колонки
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["day_of_year"] = daily["date"].dt.day_of_year
    daily["weekday"] = daily["date"].dt.weekday   # 0=пн, 6=вс

    os.makedirs("data", exist_ok=True)
    daily.to_csv(OUTPUT_CSV, index=False)

    print(f"\nПервые 10 дней с наибольшим количеством отчётов:")
    print(daily.nlargest(10, "report_count")[["date", "report_count", "year"]].to_string(index=False))
    print(f"\nСохранено: {OUTPUT_CSV} ({len(daily)} уникальных дат)")

    # Краткая статистика по годам
    print("\nОтчёты по годам:")
    yearly = daily.groupby("year")["report_count"].sum().reset_index()
    print(yearly.to_string(index=False))


if __name__ == "__main__":
    aggregate()
