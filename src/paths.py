"""
Утилита для подмены путей в скриптах.
Каждый скрипт вызывает setup_paths(city_config) в начале main().
"""

from src.config import CityConfig


def setup_paths(city: CityConfig) -> dict:
    """Возвращает словарь путей для города."""
    return {
        "raw_posts_json":       city.path("raw_posts.json"),
        "raw_posts_csv":        city.path("raw_posts.csv"),
        "checkpoint":           city.path("checkpoint.json"),
        "posts_with_dates":     city.path("posts_with_dates.csv"),
        "photo_species":        city.path("photo_species.csv"),
        "photo_checkpoint":     city.path("photo_species_checkpoint.json"),
        "no_mushroom_ids":      city.path("no_mushroom_ids.json"),
        "daily_counts":         city.path("daily_counts.csv"),
        "weather_raw":          city.path("weather_raw.csv"),
        "weather_features":     city.path("weather_features.csv"),
        "panel":                city.path("panel.csv"),
        "panel_predictions":    city.path("panel_predictions.csv"),
        "selected_features":    city.path("selected_features.json"),
    }
