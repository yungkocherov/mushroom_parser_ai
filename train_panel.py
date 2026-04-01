"""
Панельная модель: LightGBM для предсказания количества грибов по видам.

Вход:  data/panel.csv (дата × вид × погодные фичи)
Выход: data/panel_model.lgb, data/panel_predictions.csv

Структура данных:
  Каждая строка = один день × один вид гриба
  species — категориальная фича (LightGBM обрабатывает нативно)
  Таргет — mushroom_count (сумма грибов этого вида в этот день по фото)

Запуск: python train_panel.py
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PANEL_CSV    = "data/panel.csv"
OUTPUT_MODEL = "data/panel_model.lgb"
OUTPUT_PREDS = "data/panel_predictions.csv"

TARGET = "mushroom_count"
TEST_YEAR = 2025

NON_FEATURES = {
    "date", "mushroom_count",
    # Служебные из aggregate
    "report_count", "mushroom_index", "mushroom_index_sm",
    "audience_scale", "avg_likes", "total_photos", "avg_views",
    "median_views", "audience_proxy", "vk_trend", "weekday_scale",
    # Человеческий фактор
    "year", "weekday", "is_weekend", "is_holiday", "is_day_off", "long_weekend",
    "day_of_month",
    "is_around_20th_may", "is_around_20th_jun", "is_around_20th_jul",
    "is_around_20th_aug", "is_around_20th_sep", "is_peak_window",
    "days_to_aug20", "days_to_sep20", "days_to_peak", "season_gauss",
}


def get_feature_cols(df):
    return [c for c in df.columns
            if c not in NON_FEATURES and c != "date" and df[c].dtype != object
            or c == "species"]  # species — категориальная


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL_CSV, parse_dates=["date"])

    # species → категориальный тип (LightGBM обрабатывает нативно)
    df["species"] = df["species"].astype("category")

    return df


def train():
    df = load_panel()

    feature_cols = get_feature_cols(df)
    cat_features = ["species"]

    train_df = df[df["date"].dt.year < TEST_YEAR]
    test_df  = df[df["date"].dt.year == TEST_YEAR]

    X_tr, y_tr = train_df[feature_cols], train_df[TARGET]
    X_te, y_te = test_df[feature_cols],  test_df[TARGET]

    print(f"Panel model")
    print(f"  Train: {len(train_df)} rows ({train_df['date'].dt.year.min()}-{train_df['date'].dt.year.max()})")
    print(f"  Test:  {len(test_df)} rows ({TEST_YEAR})")
    print(f"  Features: {len(feature_cols)} (incl. species as categorical)")
    print(f"  Species: {df['species'].cat.categories.tolist()}")

    model = lgb.LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        num_leaves=63,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5,
        verbose=-1,
        random_state=42,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(200)],
        categorical_feature=cat_features,
    )

    preds_test  = model.predict(X_te).clip(0)
    preds_train = model.predict(X_tr).clip(0)

    # Метрики — общие
    rmse = np.sqrt(mean_squared_error(y_te, preds_test))
    mae  = mean_absolute_error(y_te, preds_test)
    r2   = r2_score(y_te, preds_test)
    r2_tr = r2_score(y_tr, preds_train)

    print(f"\n{'='*50}")
    print(f"Overall:  RMSE={rmse:.3f}, MAE={mae:.3f}")
    print(f"          R2 test={r2:.3f}, R2 train={r2_tr:.3f}")

    # Метрики по видам
    test_df = test_df.copy()
    test_df["predicted"] = preds_test
    print(f"\nR2 по видам (тест):")
    for sp in sorted(df["species"].cat.categories):
        sp_mask = test_df["species"] == sp
        if sp_mask.sum() == 0:
            continue
        sp_r2 = r2_score(test_df.loc[sp_mask, TARGET], test_df.loc[sp_mask, "predicted"])
        sp_mean = test_df.loc[sp_mask, TARGET].mean()
        print(f"  {sp:25s} R2={sp_r2:.3f}  mean={sp_mean:.1f}")

    # Сохраняем
    model.booster_.save_model(OUTPUT_MODEL)
    print(f"\nМодель: {OUTPUT_MODEL}")

    result = test_df[["date", "species", TARGET]].copy()
    result["predicted"] = preds_test.round(1)
    result.to_csv(OUTPUT_PREDS, index=False)
    print(f"Предсказания: {OUTPUT_PREDS}")

    # Feature importance
    importance = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print(f"\nТоп-20 фич:")
    print(importance.head(20).to_string())

    return model, importance, result


if __name__ == "__main__":
    train()
