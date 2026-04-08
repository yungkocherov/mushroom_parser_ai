"""
5 панельных моделей — по одной на каждую группу грибов.
Каждая модель обучается только на своём сезоне.

Вход:  data/panel.csv
Выход: data/panel_model_{group}.lgb, data/panel_predictions.csv

Walk-forward CV + Optuna для каждой группы отдельно.

Запуск: python train_panel.py
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

PANEL_CSV    = None
OUTPUT_DIR   = None
OUTPUT_PREDS = None

TARGET = "mushroom_count"
TEST_YEAR = 2025

# Сезон каждой группы — обучаем только на этих месяцах
GROUP_SEASONS = None

# Фичи, исключённые из модели
NON_FEATURES = {
    "date", "mushroom_count", "species",
    # Служебные
    "report_count", "mushroom_index", "mushroom_index_sm",
    "audience_scale", "avg_likes", "total_photos", "avg_views",
    "median_views", "audience_proxy", "vk_trend", "weekday_scale",
    # Человеческий фактор
    "year", "weekday", "is_weekend", "is_holiday", "is_day_off", "long_weekend",
    "day_of_month", "month", "week_of_year",
    "is_around_20th_may", "is_around_20th_jun", "is_around_20th_jul",
    "is_around_20th_aug", "is_around_20th_sep", "is_peak_window",
    "days_to_aug20", "days_to_sep20", "days_to_peak", "season_gauss",
    "sin_doy1", "cos_doy1", "sin_doy2", "cos_doy2", "sin_doy3", "cos_doy3",
    "sin_doy4", "cos_doy4", "sin_doy5", "cos_doy5",
}

CV_YEARS = None
N_TRIALS = None

SELECTED_FEATURES_FILE = None


def get_feature_cols(df, group=None):
    """Возвращает фичи. Если есть selected_features.json — берёт оттуда для группы."""
    if group and os.path.exists(SELECTED_FEATURES_FILE):
        import json
        with open(SELECTED_FEATURES_FILE, encoding="utf-8") as f:
            selected = json.load(f)
        if group in selected:
            return [c for c in selected[group] if c in df.columns]
    # Fallback: все фичи
    return [c for c in df.columns
            if c not in NON_FEATURES and c != "date" and df[c].dtype != object]


def load_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL_CSV, parse_dates=["date"])
    return df


def eval_params(df, params, feature_cols, cv_years=None):
    """Walk-forward CV для одной группы."""
    if cv_years is None:
        cv_years = CV_YEARS

    folds = []
    for test_year in cv_years:
        train_df = df[df["date"].dt.year < test_year]
        test_df  = df[df["date"].dt.year == test_year]
        if len(train_df) < 50 or len(test_df) < 10:
            continue

        X_tr, y_tr = train_df[feature_cols], train_df[TARGET]
        X_te, y_te = test_df[feature_cols],  test_df[TARGET]

        model = lgb.LGBMRegressor(**params, verbose=-1, random_state=42)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_te, y_te)],
                  callbacks=[lgb.early_stopping(30, verbose=False)])

        preds = model.predict(X_te).clip(0)
        folds.append({
            "year": test_year,
            "rmse": np.sqrt(mean_squared_error(y_te, preds)),
            "mae":  mean_absolute_error(y_te, preds),
            "r2":   r2_score(y_te, preds) if y_te.std() > 0 else 0,
        })

    return pd.DataFrame(folds)


def train_group(df_full, group, season_months, feature_cols):
    """Обучает одну модель для одной группы."""
    print(f"\n{'='*60}")
    print(f"  Группа: {group}")
    print(f"  Сезон: месяцы {season_months}")

    # Фильтруем: только эта группа + только сезонные месяцы
    df = df_full[
        (df_full["species"] == group) &
        (df_full["date"].dt.month.isin(season_months))
    ].copy()

    print(f"  Строк: {len(df)}, ненулевых: {(df[TARGET] > 0).sum()}")

    if len(df) < 100:
        print(f"  Пропускаем — мало данных")
        return None, None, None

    # Baseline — простая модель, устойчивая к переобучению
    BASE_PARAMS = dict(
        n_estimators=300, learning_rate=0.05,
        max_depth=4, num_leaves=15,
        min_child_samples=50, subsample=0.7, colsample_bytree=0.7,
        reg_alpha=1.0, reg_lambda=1.0,
    )

    baseline_cv = eval_params(df, BASE_PARAMS, feature_cols)
    if len(baseline_cv) == 0:
        print(f"  Пропускаем — недостаточно данных для CV")
        return None, None, None
    print(f"  Baseline: RMSE={baseline_cv['rmse'].mean():.3f}, R2={baseline_cv['r2'].mean():.3f}")

    # Optuna — ограниченные диапазоны для устойчивых моделей
    def objective(trial):
        params = dict(
            n_estimators      = trial.suggest_int("n_estimators", 100, 1000),
            learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            num_leaves        = trial.suggest_int("num_leaves", 7, 31),
            max_depth         = trial.suggest_int("max_depth", 3, 6),
            min_child_samples = trial.suggest_int("min_child_samples", 30, 100),
            min_split_gain    = trial.suggest_float("min_split_gain", 0.1, 2.0),
            subsample         = trial.suggest_float("subsample", 0.5, 0.8),
            subsample_freq    = trial.suggest_int("subsample_freq", 1, 5),
            colsample_bytree  = trial.suggest_float("colsample_bytree", 0.5, 0.8),
            reg_alpha         = trial.suggest_float("reg_alpha", 0.5, 20.0, log=True),
            reg_lambda        = trial.suggest_float("reg_lambda", 0.5, 20.0, log=True),
        )
        folds = eval_params(df, params, feature_cols)
        if len(folds) == 0:
            return float("inf")
        return folds["rmse"].mean()

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=15),
    )
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_params = study.best_params
    best_cv = eval_params(df, best_params, feature_cols)
    print(f"  Optuna ({N_TRIALS} trials): RMSE={best_cv['rmse'].mean():.3f}, R2={best_cv['r2'].mean():.3f}")

    # Финальная модель
    train_df = df[df["date"].dt.year < TEST_YEAR]
    test_df  = df[df["date"].dt.year == TEST_YEAR]

    X_tr, y_tr = train_df[feature_cols], train_df[TARGET]
    X_te, y_te = test_df[feature_cols],  test_df[TARGET]

    final_model = lgb.LGBMRegressor(**best_params, verbose=-1, random_state=42)
    final_model.fit(
        X_tr, y_tr,
        eval_set=[(X_te, y_te)],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    preds_test  = final_model.predict(X_te).clip(0)
    preds_train = final_model.predict(X_tr).clip(0)

    r2_test = r2_score(y_te, preds_test) if y_te.std() > 0 else 0
    r2_train = r2_score(y_tr, preds_train) if y_tr.std() > 0 else 0
    rmse_test = np.sqrt(mean_squared_error(y_te, preds_test))
    mae_test = mean_absolute_error(y_te, preds_test)

    print(f"  Финал: R2 test={r2_test:.3f}, R2 train={r2_train:.3f}, RMSE={rmse_test:.3f}, MAE={mae_test:.3f}")

    # Сохраняем модель
    model_path = os.path.join(OUTPUT_DIR, f"panel_model_{group}.lgb")
    final_model.booster_.save_model(model_path)
    print(f"  Модель: {model_path}")

    # Feature importance
    importance = pd.Series(
        final_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print(f"  Топ-10 фич:")
    for feat, imp in importance.head(10).items():
        print(f"    {feat:35s} {imp}")

    # Предсказания
    train_result = train_df[["date", TARGET]].copy()
    train_result["predicted"] = preds_train
    train_result["species"] = group

    test_result = test_df[["date", TARGET]].copy()
    test_result["predicted"] = preds_test
    test_result["species"] = group

    result = pd.concat([train_result, test_result], ignore_index=True)

    return final_model, importance, result


def main(city_config=None, app_config=None):
    global PANEL_CSV, OUTPUT_DIR, OUTPUT_PREDS, SELECTED_FEATURES_FILE
    global GROUP_SEASONS, CV_YEARS, N_TRIALS, TEST_YEAR

    if city_config:
        PANEL_CSV              = city_config.path("panel.csv")
        OUTPUT_DIR             = str(city_config.data_dir)
        OUTPUT_PREDS           = city_config.path("panel_predictions.csv")
        SELECTED_FEATURES_FILE = city_config.path("selected_features.json")
    else:
        PANEL_CSV              = PANEL_CSV or "data/panel.csv"
        OUTPUT_DIR             = OUTPUT_DIR or "data"
        OUTPUT_PREDS           = OUTPUT_PREDS or "data/panel_predictions.csv"
        SELECTED_FEATURES_FILE = SELECTED_FEATURES_FILE or "data/selected_features.json"

    if app_config:
        GROUP_SEASONS = app_config.group_seasons or GROUP_SEASONS
        CV_YEARS      = app_config.cv_years or CV_YEARS
        N_TRIALS      = app_config.optuna_trials or N_TRIALS
        TEST_YEAR     = app_config.test_year or TEST_YEAR

    # Fallbacks for standalone execution
    GROUP_SEASONS = GROUP_SEASONS or {
        "болетовые":   list(range(6, 11)),
        "лисичковые":  list(range(6, 12)),
        "весенние":    [4, 5],
        "опята":       list(range(6, 11)),
    }
    CV_YEARS = CV_YEARS or [2023, 2024, 2025]
    N_TRIALS = N_TRIALS or 150
    TEST_YEAR = TEST_YEAR or 2025

    train()


def train():
    df_full = load_panel()

    print(f"Панель: {len(df_full)} строк")
    print(f"Группы: {list(GROUP_SEASONS.keys())}")
    print(f"CV годы: {CV_YEARS}, тест: {TEST_YEAR}")
    print(f"Optuna: {N_TRIALS} trials на группу")

    all_results = []
    all_models = {}
    all_importances = {}

    for group, season_months in GROUP_SEASONS.items():
        feature_cols = get_feature_cols(df_full, group=group)
        print(f"\n[{group}] Фич: {len(feature_cols)}")
        model, importance, result = train_group(df_full, group, season_months, feature_cols)
        if model is not None:
            all_models[group] = model
            all_importances[group] = importance
            all_results.append(result)

    # Сохраняем объединённые предсказания
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(OUTPUT_PREDS, index=False)
        print(f"\nПредсказания: {OUTPUT_PREDS} ({len(combined)} строк)")

    # Итоговая таблица
    print(f"\n{'='*60}")
    print("ИТОГО:")
    for group in GROUP_SEASONS:
        if group in all_models:
            model = all_models[group]
            # Пересчитываем метрики на тесте
            df_g = df_full[
                (df_full["species"] == group) &
                (df_full["date"].dt.month.isin(GROUP_SEASONS[group]))
            ]
            test_g = df_g[df_g["date"].dt.year == TEST_YEAR]
            if len(test_g) > 0:
                g_feature_cols = get_feature_cols(df_full, group=group)
                X_te = test_g[g_feature_cols]
                y_te = test_g[TARGET]
                preds = model.predict(X_te).clip(0)
                r2 = r2_score(y_te, preds) if y_te.std() > 0 else 0
                print(f"  {group:30s} R2={r2:.3f}")

    return all_models, all_importances


if __name__ == "__main__":
    main()
