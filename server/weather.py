# -*- coding: utf-8 -*-
"""Weather answer from Open-Meteo: free, keyless, JSON, no scraping.

Measured problem: a weather question went through DuckDuckGo, and when
DDG served junk (bot-suspect pages) the answer was "Не нашёл" or a
citation dump from an unrelated Wikipedia page. Weather is the most
common live-data ask, so it gets a direct provider with a search
fallback in main.py.
"""
from __future__ import annotations

import json
import re
import urllib.parse

from .web import _get_text

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FCST_URL = "https://api.open-meteo.com/v1/forecast"

WEATHER_RE = re.compile(
    r"погод|температур|осадк|ветер|дожд|снег|мороз|жар[аоы]|зонтик|"
    r"weather|forecast|temperature",
    re.IGNORECASE,
)
CITY_RE = re.compile(
    r"\b(?:в|во|по|на)\s+([А-ЯЁа-яёA-Za-z][А-ЯЁа-яёA-Za-z\-\s]{1,40})",
)

# Words that follow a city name in a question and are never part of it.
_NOT_CITY = {
    "сегодня", "завтра", "послезавтра", "сейчас", "утром", "вечером", "ночью",
    "днём", "днем", "неделю", "недель", "выходные", "выходных", "будет",
    "ожидается", "ли", "на", "у", "и", "во", "за", "окном", "эту", "этот",
    "ближайшие", "какая", "какой", "сколько", "градусов",
}

# First words of two-word city names (Нижний Новгород, Великий Новгород).
_MULTIWORD = {
    "нижний", "нижнем", "нижняя", "нижнюю", "великий", "великом", "великие",
    "старый", "старом", "новый", "новом", "санкт", "петрозаводск", "сергиев",
    "северный", "северном", "южный", "южном", "восточный", "восточном",
    "западный", "западном", "красная", "красный", "белая", "белый", "полярный",
    "советский", "советской", "петропавловск", "усть", "усть-каменск",
    "светлый", "светлом", "правда", "горький",
}

# Capitalised words that open a weather question and are not a city.
_NOT_NAME = {
    "погода", "погоду", "погоды", "прогноз", "температура", "температуру",
    "какая", "какой", "какие", "сколько", "скажи", "покажи", "посмотри",
    "узнай", "найди", "дай", "будет", "завтра", "сегодня", "сейчас",
}

# WMO weather interpretation codes -> short Russian descriptions.
_WMO = {
    0: "ясно", 1: "преимущественно ясно", 2: "переменная облачность",
    3: "пасмурно", 45: "туман", 48: "изморозь и туман",
    51: "слабая морось", 53: "морось", 55: "сильная морось",
    56: "ледяная морось", 57: "сильная ледяная морось",
    61: "слабый дождь", 63: "дождь", 65: "сильный дождь",
    66: "ледяной дождь", 67: "сильный ледяной дождь",
    71: "слабый снег", 73: "снег", 75: "сильный снег", 77: "снежные зёрна",
    80: "слабые ливни", 81: "ливни", 82: "сильные ливни",
    85: "снежные ливни", 86: "сильные снежные ливни",
    95: "гроза", 96: "гроза с градом", 99: "сильная гроза с градом",
}


# Nominative forms of inflected first words («в нижнем» → «нижний»).
_NOM = {
    "нижний": "нижний", "нижнем": "нижний", "нижняя": "нижняя",
    "нижнюю": "нижняя", "великий": "великий", "великом": "великий",
    "великие": "великие", "великая": "великая",
    "старый": "старый", "старом": "старый", "новый": "новый",
    "новом": "новый", "северный": "северный", "северном": "северный",
    "южный": "южный", "южном": "южный", "восточный": "восточный",
    "восточном": "восточный", "западный": "западный",
    "западном": "западный", "красная": "красная", "красный": "красный",
    "белая": "белая", "белый": "белый", "полярный": "полярный",
    "горький": "горький", "горьком": "горький", "светлый": "светлый",
    "светлом": "светлый", "советский": "советский",
    "советской": "советская", "золотое": "золотое", "санкт": "санкт",
}


def is_weather(question: str) -> bool:
    return bool(WEATHER_RE.search(question or ""))


def _join_name(tokens: list[str]) -> str | None:
    """City from the words after a preposition, in the case they arrived in."""
    tokens = [t.strip(".,!?;:") for t in tokens if t.strip(".,!?;:")]
    if not tokens or tokens[0].lower() in _NOT_CITY:
        return None
    name = tokens[0]
    if len(tokens) > 1 and tokens[0].lower() in _MULTIWORD \
            and tokens[1].lower() not in _NOT_CITY:
        name += " " + tokens[1]
    return name


def _city(question: str) -> str:
    """City name as written in the question; Москва when none is named."""
    for m in CITY_RE.finditer(question):
        name = _join_name(m.group(1).split())
        if name:
            return name
    # No preposition ("погода Москва"): fall back to a capitalised word
    # that is not a weather word of its own.
    for m in re.finditer(r"\b[А-ЯЁA-Z][а-яёa-z\-]{2,20}\b", question):
        if m.group(0).lower() not in _NOT_NAME:
            return m.group(0)
    return "Москва"


def _candidates(city: str) -> list[str]:
    """Search forms for an inflected city name, best guess first.

    Questions arrive in any case («в казани», «в нижнем новгороде»);
    the geocoder prefix-matches nominative forms, so each token gets
    a de-inflected variant.
    """
    tokens = city.split()
    if not tokens:
        return [city]
    if len(tokens) == 1:
        token = tokens[0]
        # trimmed stems first: an inflected form prefix-matches only
        # villages («москве» → Москверо), never the real city
        out = []
        nominative = _NOM.get(token.lower())
        if nominative:
            out.append(nominative)
        for cut in (1, 2):
            if len(token) > cut + 3:
                out.append(token[:-cut])
        out.append(city)
        return out
    first, last = tokens[0], tokens[-1]
    nominative_first = _NOM.get(first.lower(), first)
    trimmed_last = last
    for suffix in ("е", "и", "у", "а", "я"):
        if trimmed_last.endswith(suffix) and len(trimmed_last) > 4:
            trimmed_last = trimmed_last[:-1]
            break
    out = [f"{nominative_first} {trimmed_last}", trimmed_last, city]
    return [c for c in out if c.strip()]


def _geocode(city: str) -> tuple[float, float, str] | None:
    """Resolve a city to coordinates, ranked by population.

    Prefix matching alone returns whatever matches first — «москве»
    resolved to a village called Москверо. Among the candidates the
    most populous place is the intended one in virtually every case.
    """
    for name in _candidates(city):
        url = GEO_URL + "?" + urllib.parse.urlencode(
            {"name": name, "count": "5", "language": "ru", "format": "json"})
        body = _get_text(url)
        if not body:
            continue
        try:
            results = json.loads(body).get("results") or []
        except (json.JSONDecodeError, AttributeError):
            continue
        if not results:
            continue
        best = max(results, key=lambda item: int(item.get("population") or 0))
        return (
            float(best["latitude"]),
            float(best["longitude"]),
            str(best.get("name", name)),
        )
    return None


def weather_text(question: str) -> tuple[str, str] | None:
    """(facts, source) for a weather question, or None on failure."""
    city = _city(question)
    geo = _geocode(city)
    if geo is None:
        return None
    lat, lon, name = geo
    url = FCST_URL + "?" + urllib.parse.urlencode({
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
        "current": ",".join((
            "temperature_2m", "relative_humidity_2m", "apparent_temperature",
            "precipitation", "weather_code", "wind_speed_10m",
            "pressure_msl",
        )),
        "timezone": "auto",
    })
    body = _get_text(url)
    if not body:
        return None
    try:
        cur = json.loads(body)["current"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None
    code = int(cur.get("weather_code", 0))
    desc = _WMO.get(code, f"код погоды {code}")
    parts = [
        f"Сейчас в {name}: {round(float(cur['temperature_2m']))}°C "
        f"(ощущается как {round(float(cur['apparent_temperature']))}°C), {desc}.",
        f"Ветер {round(float(cur['wind_speed_10m']))} м/с, "
        f"влажность {round(float(cur['relative_humidity_2m']))}%, "
        f"давление {round(float(cur['pressure_msl']))} гПа, "
        f"осадки {float(cur.get('precipitation', 0.0)):.1f} мм.",
    ]
    return "\n".join(parts), "https://open-meteo.com/"
