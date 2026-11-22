# -*- coding: utf-8 -*-
"""Bulk-fetch survival/medicine/radio/electronics articles from Wikipedia.

Transport: urllib only — on this machine httpx gets 202/403 from Wikimedia,
urllib gets clean 200 (verified earlier in server/web.py work).
One title per request (full-text extracts limit), 0.25s between calls,
both en and ru editions, output one .txt per article under data/corpus/survival/.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("data/corpus/survival")
OUT.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "extra-code-kb/1.0 (local research; contact: local)"}

TOPICS = {
    "first_aid": [
        "First aid", "Cardiopulmonary resuscitation", "Bleeding", "Tourniquet",
        "Wound", "Bandage", "Fracture", "Splint", "Burn", "Sunburn",
        "Hypothermia", "Hyperthermia", "Frostbite", "Dehydration",
        "Heat stroke", "Choking", "Recovery position", "ABC (medicine)",
        "Anaphylaxis", "EpiPen", "Snakebite", "Insect sting", "Tick bite",
        "Food poisoning", "Salmonellosis", "Tetanus", "Rabies",
        "Drowning", "Electric shock", "Smoke inhalation", "Carbon monoxide poisoning",
    ],
    "medicine": [
        "Medicine", "Symptom", "Diagnosis", "Differential diagnosis",
        "Infection", "Sepsis", "Antibiotic", "Penicillin", "Amoxicillin",
        "Analgesic", "Paracetamol", "Ibuprofen", "Aspirin", "Opium",
        "Willow bark", "Medicinal plants", "Herbalism", "Phytotherapy",
        "Oral rehydration therapy", "Electrolyte", "Activated carbon",
        "Honey", "Antiseptic", "Disinfection", "Sterilization (microbiology)",
        "Bandage#Improvised", "Suture", "Cauterization", "Toilet paper",
        "Dysentery", "Cholera", "Typhoid fever", "Malaria", "Influenza",
        "Common cold", "Pneumonia", "Bronchitis", "Gastritis", "Ulcer (dermatology)",
        "Dental emergency", "Toothache", "Childbirth", "Obstetrics",
        "Mental disorder", "Panic attack", "Post-traumatic stress disorder",
        "Triage", "Field hospital", "Improvised explosive device",
    ],
    "survival": [
        "Survival skills", "Survival kit", "Bushcraft", "Wilderness medicine",
        "Shelter (building)", "Lean-to", "Debris hut", "Fire making",
        "Fire piston", "Bow drill", "Tinder", "Water purification",
        "Solar water disinfection", "Boiling", "Sand filter",
        "Foraging", "Edible plants", "Mushroom hunting", "Trapping (animal)",
        "Snare trap", "Fishing techniques", "Hunting", "Skinning",
        "Navigation", "Orienteering", "Compass", "Topographic map",
        "Celestial navigation", "Finding north", "Knot", "List of knots",
        "Bowline", "Taut-line hitch", "Paracord", "Camouflage",
        "Tracking (hunting)", "Weather forecasting", "Cloud", "Barometer",
        "Winter survival", "Desert survival", "Jungle survival",
        " avalanche", "Quicksand", "Wilderness Emergency Medical Technician",
        "Prepper", "Disaster preparedness", "Bug-out bag", "Food storage",
        "Canning", "Drying (food)", "Smoking (cooking)", "Salting (food)",
        "Water storage", "Off-grid", "Homesteading",
    ],
    "radio_comms": [
        "Radio", "Shortwave radio", "Amateur radio", "Ham radio",
        "Crystal radio", "Radio receiver", "Transceiver", "Walkie-talkie",
        "Antenna (radio)", "Dipole antenna", "Whip antenna", "Loop antenna",
        "Antenna tuner", "Feed line", "Coaxial cable", "SWR meter",
        "Morse code", "Morse code abbreviations", "Q code", "Phonetic alphabet",
        "Amateur radio license", "Emergency communication",
        "Radio propagation", "Ionosphere", "Frequency", "Wavelength",
        "AM broadcasting", "FM broadcasting", "Single-sideband modulation",
        "Citizens band radio", "GMRS", "MURS", "FRS (radio)",
        "Satellite phone", "Signal lamp", "Semaphore (communication)",
        "Mirror signalling", "Smoke signal", "Whistle signal",
        "Electromagnetic pulse", "Faraday cage", "Solar storm",
        "Battery (electricity)", "Rechargeable battery", "Solar panel",
        "Hand-crank radio", "Two-way radio", "Radio scanner",
    ],
    "electronics": [
        "Electronics", "Soldering", "Solder", "Soldering iron",
        "Multimeter", "Oscilloscope", "Breadboard", "Printed circuit board",
        "Circuit diagram", "Electronic component", "Resistor", "Capacitor",
        "Inductor", "Transistor", "Diode", "LED", "Integrated circuit",
        "Operational amplifier", "Voltage regulator", "Transformer",
        "Rectifier", "Power supply", "Battery charger", "Alternating current",
        "Direct current", "Ohm's law", "Series and parallel circuits",
        "Short circuit", "Fuse (electrical)", "Wire", "Copper wire",
        "Electrical wiring", "Electrical grounding", "Continuity test",
        "Electronic waste", "Salvaging", "Jumper wire", "Alligator clip",
        "Heat gun", "Desoldering", "Electrical tape", "Insulation",
        "Electric motor repair", "Generator", "Alternator",
        "Inverter (electrical)", "UPS (electricity)",
    ],
}

RU_TOPICS = [
    "Первая помощь", "Сердечно-лёгочная реанимация", "Кровотечение", "Жгут",
    "Рана", "Перелом", "Ожог", "Гипотермия", "Тепловой удар", "Обморожение",
    "Обезвоживание", "Утопление", "Отравление", "Столбняк", "Бешенство",
    "Анафилаксия", "Укус змеи", "Антибиотики", "Парацетамол", "Ибупрофен",
    "Аспирин", "Лекарственные растения", "Фитотерапия", "Мёд",
    "Регидратация", "Активированный уголь", "Антисептик", "Дезинфекция",
    "Холера", "Дизентерия", "Малярия", "Грипп", "Пневмония", "Зубная боль",
    "Роды", "Триаж", "Выживание", "Набор выживания", "Бушкрафт",
    "Костёр", "Добывание огня", "Очистка воды", "Ловушка", "Охота",
    "Рыбалка", "Ориентирование", "Компас", "Топографическая карта",
    "Узел", "Список узлов", "Булинь", "Паракорд", "Маскировка",
    "Прогноз погоды", "Облака", "Багаж выживания", "Консервирование",
    "Копчение", "Засолка", "Автономное существование",
    "Радио", "Коротковолновая связь", "Радиолюбительство", "Радиоприёмник",
    "Детекторный приёмник", "Радиостанция", "Антенна", "Дипольная антенна",
    "Коаксиальный кабель", "Азбука Морзе", "Q-код", "ФОНАРНАЯ азбука",
    "Распространение радиоволн", "Ионосфера", "Однополосная модуляция",
    "Си-Би", "Спутниковый телефон", "Электромагнитный импульс", "Клетка Фарадея",
    "Электроника", "Пайка", "Припой", "Мультиметр", "Осциллограф",
    "Макетная плата", "Печатная плата", "Принципиальная схема",
    "Резистор", "Конденсатор", "Катушка индуктивности", "Транзистор",
    "Диод", "Светодиод", "Интегральная схема", "Стабилизатор напряжения",
    "Трансформатор", "Выпрямитель", "Блок питания", "Аккумулятор",
    "Солнечная батарея", "Закон Ома", "Короткое замыкание", "Предохранитель",
    "Провод", "Заземление", "Паяльная станция", "Портативная пайка",
]


def fetch_extract(lang: str, title: str) -> str | None:
    api = f"https://{lang}.wikipedia.org/w/api.php"
    params = {
        "action": "query", "prop": "extracts", "explaintext": "1",
        "redirects": "1", "format": "json", "titles": title,
    }
    url = api + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            body = json.loads(r.read().decode("utf-8"))
    except Exception as exc:
        print(f"  ERR {lang}:{title}: {exc}")
        return None
    pages = body.get("query", {}).get("pages", {})
    for page in pages.values():
        text = page.get("extract")
        if text and len(text) > 400:
            return text
    return None


def slug(title: str) -> str:
    s = re.sub(r"[^0-9A-Za-zА-Яа-яЁё]+", "_", title.strip())
    return s.strip("_")[:80]


def main() -> None:
    ok = fail = 0
    for group, titles in TOPICS.items():
        for title in titles:
            t = title.strip()
            if not t or t.startswith("#"):
                continue
            out = OUT / f"en_{group}_{slug(t)}.txt"
            if out.exists() and out.stat().st_size > 1000:
                ok += 1
                continue
            text = fetch_extract("en", t)
            time.sleep(0.25)
            if text:
                out.write_text(text, encoding="utf-8")
                ok += 1
            else:
                fail += 1
        print(f"done en group {group}: running ok={ok} fail={fail}")
    for title in RU_TOPICS:
        t = title.strip()
        if not t:
            continue
        out = OUT / f"ru_{slug(t)}.txt"
        if out.exists() and out.stat().st_size > 1000:
            ok += 1
            continue
        text = fetch_extract("ru", t)
        time.sleep(0.25)
        if text:
            out.write_text(text, encoding="utf-8")
            ok += 1
        else:
            fail += 1
    total = sum(1 for _ in OUT.glob("*.txt"))
    mb = sum(p.stat().st_size for p in OUT.glob("*.txt")) / 1e6
    print(json.dumps({"fetched_ok": ok, "failed": fail,
                      "files_on_disk": total, "mb": round(mb, 1)}))


if __name__ == "__main__":
    main()
