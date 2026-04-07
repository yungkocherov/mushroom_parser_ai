"""
Загрузка конфигурации из config.yaml.
Все скрипты используют этот модуль для путей и настроек.
"""

import os
import yaml
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


@dataclass
class CityConfig:
    key: str                    # "spb", "moscow"
    vk_group: str               # "grib_spb"
    city_name: str              # "Санкт-Петербург"
    lat: float                  # 59.9343
    lon: float                  # 30.3351
    timezone: str               # "Europe/Moscow"
    years_back: int             # 8
    data_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = PROJECT_ROOT / "data" / self.key
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def path(self, filename: str) -> str:
        """Возвращает полный путь к файлу данных: data/{city}/{filename}"""
        return str(self.data_dir / filename)


@dataclass
class AppConfig:
    cities: dict[str, CityConfig]
    lm_studio_url: str
    lm_studio_model: str
    optuna_trials: int
    cv_years: list[int]
    test_year: int
    n_workers: int
    group_seasons: dict[str, list[int]]


def load_config() -> AppConfig:
    """Загружает config.yaml и возвращает AppConfig."""
    with open(CONFIG_PATH, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    defaults = raw.get("defaults", {})
    cities = {}
    for key, city_raw in raw.get("cities", {}).items():
        cities[key] = CityConfig(
            key=key,
            vk_group=city_raw["vk_group"],
            city_name=city_raw["city_name"],
            lat=city_raw["lat"],
            lon=city_raw["lon"],
            timezone=city_raw.get("timezone", "Europe/Moscow"),
            years_back=city_raw.get("years_back", 6),
        )

    return AppConfig(
        cities=cities,
        lm_studio_url=defaults.get("lm_studio_url", "http://localhost:1234/v1/chat/completions"),
        lm_studio_model=defaults.get("lm_studio_model", "google/gemma-3-12b"),
        optuna_trials=defaults.get("optuna_trials", 150),
        cv_years=defaults.get("cv_years", [2023, 2024, 2025]),
        test_year=defaults.get("test_year", 2025),
        n_workers=defaults.get("n_workers", 4),
        group_seasons=raw.get("group_seasons", {}),
    )


def get_city(city_key: str = "spb") -> CityConfig:
    """Быстрый доступ к конфигу города."""
    config = load_config()
    if city_key not in config.cities:
        raise ValueError(f"Город '{city_key}' не найден в config.yaml. Доступны: {list(config.cities.keys())}")
    return config.cities[city_key]
