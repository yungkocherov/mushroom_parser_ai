"""
Запуск пайплайна для одного или всех городов.

Использование:
  python run_pipeline.py                          # все города, все шаги
  python run_pipeline.py --city spb               # один город
  python run_pipeline.py --city spb --step train   # один шаг
  python run_pipeline.py --city spb --from weather # от шага и далее
  python run_pipeline.py --list                    # список шагов
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.config import load_config

STEPS = [
    ("collect",    "Сбор постов из VK",              "src.collect_posts"),
    ("dates",      "Извлечение дат из текста",        "src.extract_dates"),
    ("photos",     "Классификация по фото",           "src.classify_photos"),
    ("audience",   "Аудитория и daily_counts",        "src.aggregate"),
    ("weather",    "Загрузка погоды",                  "src.fetch_weather"),
    ("features",   "Генерация фич",                   "src.build_features"),
    ("panel",      "Панельная агрегация",              "src.aggregate_panel"),
    ("train",      "Обучение моделей",                 "src.train_panel"),
]


def run_step(step_name, module_path, city_config, app_config):
    """Запускает один шаг пайплайна."""
    print(f"\n{'='*60}")
    print(f"  Шаг: {step_name} | Город: {city_config.city_name}")
    print(f"{'='*60}")

    module = __import__(module_path, fromlist=["main"])
    module.main(city_config, app_config)


def main():
    parser = argparse.ArgumentParser(description="Mushroom Parser Pipeline")
    parser.add_argument("--city", type=str, help="Ключ города (spb, moscow...)")
    parser.add_argument("--step", type=str, help="Один конкретный шаг")
    parser.add_argument("--from", dest="from_step", type=str, help="Начать с этого шага")
    parser.add_argument("--list", action="store_true", help="Показать список шагов")
    args = parser.parse_args()

    if args.list:
        print("Шаги пайплайна:")
        for name, desc, _ in STEPS:
            print(f"  {name:12s} — {desc}")
        return

    config = load_config()

    # Определяем какие города обрабатывать
    if args.city:
        if args.city not in config.cities:
            print(f"Город '{args.city}' не найден. Доступны: {list(config.cities.keys())}")
            return
        cities = [config.cities[args.city]]
    else:
        cities = list(config.cities.values())

    # Определяем какие шаги запускать
    step_names = [s[0] for s in STEPS]
    if args.step:
        if args.step not in step_names:
            print(f"Шаг '{args.step}' не найден. Доступны: {step_names}")
            return
        steps_to_run = [s for s in STEPS if s[0] == args.step]
    elif args.from_step:
        if args.from_step not in step_names:
            print(f"Шаг '{args.from_step}' не найден. Доступны: {step_names}")
            return
        idx = step_names.index(args.from_step)
        steps_to_run = STEPS[idx:]
    else:
        steps_to_run = STEPS

    # Запускаем
    for city_config in cities:
        print(f"\n{'#'*60}")
        print(f"  Город: {city_config.city_name} ({city_config.key})")
        print(f"{'#'*60}")

        for step_name, step_desc, module_path in steps_to_run:
            try:
                run_step(step_name, module_path, city_config, config)
            except Exception as e:
                print(f"\n  ОШИБКА на шаге '{step_name}': {e}")
                print(f"  Продолжаю со следующим шагом...")

    print(f"\n{'='*60}")
    print("  Готово!")


if __name__ == "__main__":
    main()
