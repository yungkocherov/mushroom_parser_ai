"""
Обучение LightGBM на объединённом датасете погода + отчёты грибников.

Запуск: python train_model.py
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

WEATHER_CSV    = "data/weather_features.csv"
DAILY_CSV      = "data/daily_counts.csv"
OUTPUT_MODEL   = "data/model.lgb"
OUTPUT_PREDS   = "data/predictions.csv"

# Столбцы, которые не являются фичами
NON_FEATURES = {"date", "report_count", "avg_likes", "total_photos"}

# Грибной сезон — апрель (4) — ноябрь (11)
SEASON_MONTHS = range(4, 12)

# Тестовый период — последний год
TEST_YEAR = 2025


def load_dataset() -> pd.DataFrame:
    weather = pd.read_csv(WEATHER_CSV, parse_dates=["date"])
    daily   = pd.read_csv(DAILY_CSV,   parse_dates=["date"])

    df = weather.merge(
        daily[["date", "report_count"]],
        on="date", how="left"
    )
    df["report_count"] = df["report_count"].fillna(0).astype(int)

    # Данные с 2020 года
    df = df[df["year"] >= 2020].reset_index(drop=True)
    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in NON_FEATURES
            and c != "date" and df[c].dtype != object]


def train():
    df = load_dataset()
    feature_cols = get_feature_cols(df)

    train_df = df[df["year"] < TEST_YEAR]
    test_df  = df[df["year"] == TEST_YEAR]

    X_train, y_train = train_df[feature_cols], train_df["report_count"]
    X_test,  y_test  = test_df[feature_cols],  test_df["report_count"]

    print(f"Train: {len(train_df)} дней ({train_df['year'].min()}–{train_df['year'].max()})")
    print(f"Test:  {len(test_df)} дней ({TEST_YEAR})")
    print(f"Фич:   {len(feature_cols)}")

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)],
    )

    preds       = model.predict(X_test).clip(0)
    preds_train = model.predict(X_train).clip(0)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    r2_train = r2_score(y_train, preds_train)

    print(f"\n{'-'*40}")
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE:  {mae:.3f}")
    print(f"R2 (test):  {r2:.3f}")
    print(f"R2 (train): {r2_train:.3f}")

    # Сохраняем модель
    model.booster_.save_model(OUTPUT_MODEL)
    print(f"Модель: {OUTPUT_MODEL}")

    # Сохраняем предсказания
    result = test_df[["date", "report_count"]].copy()
    result["predicted"] = preds.round(1)
    result["error"]     = result["predicted"] - result["report_count"]
    result.to_csv(OUTPUT_PREDS, index=False)
    print(f"Предсказания: {OUTPUT_PREDS}")

    # Предсказания на трейне
    train_result = train_df[["date", "report_count"]].copy()
    train_result["predicted"] = preds_train.round(1)

    # Топ-20 важных фич
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print(f"\nТоп-20 важных фич:")
    print(importance.head(20).to_string())

    return model, importance, result, train_result


if __name__ == "__main__":
    train()
