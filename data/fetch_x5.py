# -*- coding: utf-8 -*-
"""X5 knowledge expansion: history, medicine, radio, electronics, cooking,
survival, Russian, English, Chinese, coding.

Reuses the transport from fetch_mega (batched extracts, 429 backoff,
resume-safe file naming) with a new, much larger category plan. Every
group writes en_<group>_*/ru_<group>_* files into data/corpus/survival,
which is already a KB source dir — no kb.py change needed.

Targets assume ~5.5k files on disk today and aim at ~28k total (x5).
Run from extra-code root:  python -m data.fetch_x5
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from data.fetch_mega import (
    OUT, PAUSE_S, _have, api_get, category_members, fetch_extracts, slug,
)

# group -> (total_target, en categories, ru categories)
PLAN = {
    "history": (
        4000,
        [
            "Category:Ancient history", "Category:Ancient Rome",
            "Category:Ancient Greece", "Category:Ancient Egypt",
            "Category:Middle Ages", "Category:Byzantine Empire",
            "Category:Early Modern period", "Category:Renaissance",
            "Category:Age of Discovery", "Category:Industrial Revolution",
            "Category:World War I", "Category:World War II",
            "Category:Cold War", "Category:History of Europe",
            "Category:History of Asia", "Category:History of the Americas",
            "Category:Military history", "Category:History of science",
            "Category:History of technology", "Category:Archaeology",
        ],
        [
            "Категория:История России", "Категория:История СССР",
            "Категория:Древний Рим", "Категория:Древняя Греция",
            "Категория:Древний Египет", "Категория:Средневековье",
            "Категория:Византия", "Категория:Эпоха Возрождения",
            "Категория:Первая мировая война", "Категория:Вторая мировая война",
            "Категория:Великая Отечественная война", "Категория:Холодная война",
            "Категория:История Европы", "Категория:Военная история",
            "Категория:История техники", "Категория:Древний мир",
            "Категория:Новое время", "Категория:Древняя Русь",
        ],
    ),
    "medicine": (
        4000,
        [
            "Category:Human anatomy", "Category:Physiology", "Category:Pathology",
            "Category:Toxicology", "Category:Pediatrics", "Category:Cardiology",
            "Category:Neurology", "Category:Dermatology", "Category:Endocrinology",
            "Category:Immunology", "Category:Psychiatry", "Category:Obstetrics",
            "Category:Traumatology", "Category:Medical signs",
            "Category:Medical tests", "Category:Medical treatments",
            "Category:Antibiotics", "Category:Analgesics", "Category:Vaccines",
            "Category:Poisoning", "Category:Wounds",
        ],
        [
            "Категория:Анатомия человека", "Категория:Физиология человека",
            "Категория:Токсикология", "Категория:Кардиология",
            "Категория:Неврология", "Категория:Отравления",
            "Категория:Медицинские процедуры", "Категория:Диагностика",
            "Категория:Лекарственные средства", "Категория:Заболевания по алфавиту",
            "Категория:Симптомы по алфавиту", "Категория:Первая помощь",
            "Категория:Раны", "Категория:Обморожения", "Категория:Ожоги",
        ],
    ),
    "radio": (
        2500,
        [
            "Category:Radio technology", "Category:Radio propagation",
            "Category:Radio modulation", "Category:Amateur radio equipment",
            "Category:Shortwave", "Category:Two-way radio", "Category:Radar",
            "Category:Satellite navigation", "Category:Telephony",
            "Category:Digital signal processing", "Category:Oscillators",
            "Category:Radio receivers", "Category:Radio transmitters",
            "Category:Transmission lines", "Category:Antennas (radio)",
            "Category:Radio frequency", "Category:Electronic test equipment",
        ],
        [
            "Категория:Радиоприёмники", "Категория:Радиопередатчики",
            "Категория:Антенны", "Категория:Радиолокация",
            "Категория:Радиосвязь", "Категория:Радиолюбительство",
            "Категория:Цифровая обработка сигналов", "Категория:Усилители",
            "Категория:Радиочастоты", "Категория:Радиостанции",
            "Категория:Спутниковая навигация", "Категория:Коротковолновая связь",
        ],
    ),
    "electronics": (
        2500,
        [
            "Category:Semiconductors", "Category:Transistors",
            "Category:Integrated circuits", "Category:Microcontrollers",
            "Category:Printed circuit boards", "Category:Power electronics",
            "Category:Analog circuits", "Category:Digital circuits",
            "Category:Operational amplifiers", "Category:Microprocessors",
            "Category:Sensors", "Category:Embedded systems",
            "Category:Arduino", "Category:Raspberry Pi",
            "Category:Electronic components", "Category:Electronic circuits",
            "Category:Electric power", "Category:Batteries (electricity)",
        ],
        [
            "Категория:Полупроводниковые приборы", "Категория:Транзисторы",
            "Категория:Микросхемы", "Категория:Микроконтроллеры",
            "Категория:Печатные платы", "Категория:Датчики",
            "Категория:Встраиваемые системы", "Категория:Блоки питания",
            "Категория:Электронные компоненты", "Категория:Электрические батареи",
            "Категория:Электротехника", "Категория:Усилители",
        ],
    ),
    "cooking": (
        3500,
        [
            "Category:Cooking techniques", "Category:Cooking appliances",
            "Category:Cuisines", "Category:Food preparation",
            "Category:Food safety", "Category:Food preservation",
            "Category:Breads", "Category:Dairy products", "Category:Soups",
            "Category:Salads", "Category:Sauces", "Category:Desserts",
            "Category:Beverages", "Category:Fermented foods", "Category:Spices",
            "Category:Herbs", "Category:Edible plants", "Category:Mushrooms",
            "Category:Foraging", "Category:Meat", "Category:Fish as food",
            "Category:Eggs (food)", "Category:Canning", "Category:Smoking (cooking)",
            "Category:Pickling", "Category:Baking",
        ],
        [
            "Категория:Кулинария", "Категория:Блюда по алфавиту",
            "Категория:Русская кухня", "Категория:Напитки",
            "Категория:Специи и приправы", "Категория:Съедобные грибы",
            "Категория:Несъедобные грибы", "Категория:Консервирование",
            "Категория:Пищевая безопасность", "Категория:Хлеб",
            "Категория:Супы", "Категория:Десерты", "Категория:Соусы",
            "Категория:Продукты питания", "Категория:Мясо",
            "Категория:Рыба", "Категория:Овощи", "Категория:Фрукты",
            "Категория:Заготовки", "Категория:Способы приготовления пищи",
        ],
    ),
    "surv": (
        3000,
        [
            "Category:Survival skills", "Category:Emergency management",
            "Category:Camping", "Category:Hiking equipment", "Category:Navigation",
            "Category:Knots", "Category:Firelighting", "Category:Shelters",
            "Category:Water purification", "Category:Wilderness medicine",
            "Category:Bushcraft", "Category:Survivalism",
            "Category:Weather hazards", "Category:Natural disasters",
            "Category:Self-sufficiency", "Category:Hunting", "Category:Fishing",
            "Category:Trapping (animal)", "Category:First aid",
            "Category:Civil defense", "Category:Emergency communication",
        ],
        [
            "Категория:Выживание", "Категория:Туризм",
            "Категория:Ориентирование на местности", "Категория:Узлы",
            "Категория:Костры", "Категория:Природные катастрофы",
            "Категория:Чрезвычайные ситуации", "Категория:Очистка воды",
            "Категория:Самообеспечение", "Категория:Охота",
            "Категория:Рыбалка", "Категория:Дикая природа",
            "Категория:Гражданская оборона", "Категория:Аварийные ситуации",
            "Категория:Пеший туризм", "Категория:Зимний туризм",
        ],
    ),
    "ruslang": (
        1200,
        [
            "Category:Russian grammar", "Category:Russian phonology",
            "Category:Russian vocabulary", "Category:Russian language",
            "Category:Russian orthography", "Category:Russian literature",
        ],
        [
            "Категория:Русский язык", "Категория:Грамматика русского языка",
            "Категория:Орфография", "Категория:Пунктуация",
            "Категория:Морфология русского языка", "Категория:Синтаксис русского языка",
            "Категория:Фразеология русского языка", "Категория:Пословицы и поговорки",
            "Категория:Диалекты русского языка", "Категория:Этимология",
            "Категория:Стилистика", "Категория:Лексикология",
            "Категория:Русская литература", "Категория:Орфоэпия",
        ],
    ),
    "english": (
        1200,
        [
            "Category:English grammar", "Category:English phonology",
            "Category:English vocabulary", "Category:English language",
            "Category:English idioms", "Category:English words and phrases",
            "Category:British English", "Category:American English",
            "Category:English orthography", "Category:English literature",
            "Category:Word coinage", "Category:English prefixes and suffixes",
        ],
        [
            "Категория:Английский язык", "Категория:Грамматика английского языка",
            "Категория:Английская литература", "Категория:Англицизмы",
        ],
    ),
    "chinese": (
        1200,
        [
            "Category:Chinese language", "Category:Chinese grammar",
            "Category:Chinese characters", "Category:Chinese phonology",
            "Category:Mandarin Chinese", "Category:Chinese vocabulary",
            "Category:Varieties of Chinese", "Category:Chinese written language",
            "Category:Chinese culture", "Category:History of China",
            "Category:Chinese cuisine", "Category:Chinese medicine",
        ],
        [
            "Категория:Китайский язык", "Категория:Иероглифика",
            "Категория:Пиньинь", "Категория:Диалекты китайского языка",
            "Категория:История Китая", "Категория:Китайская кухня",
            "Категория:Традиционная китайская медицина",
            "Категория:Китайская культура", "Категория:Китайские иероглифы",
        ],
    ),
    "coding": (
        2500,
        [
            "Category:Algorithms", "Category:Data structures",
            "Category:Programming paradigms", "Category:Programming language topics",
            "Category:Compiler construction", "Category:Operating systems",
            "Category:Computer networks", "Category:Databases",
            "Category:Web development", "Category:Application programming interfaces",
            "Category:Debugging", "Category:Software testing",
            "Category:Regular expressions", "Category:Object-oriented programming",
            "Category:Functional programming", "Category:Concurrency (computer science)",
            "Category:Version control", "Category:Cryptographic software",
        ],
        [
            "Категория:Алгоритмы", "Категория:Структуры данных",
            "Категория:Программирование", "Категория:Языки программирования",
            "Категория:Операционные системы", "Категория:Компьютерные сети",
            "Категория:Базы данных", "Категория:Веб-разработка",
            "Категория:Регулярные выражения", "Категория:Парадигмы программирования",
            "Категория:Криптография", "Категория:Программная инженерия",
        ],
    ),
}


def have_for(group: str) -> int:
    return len(list(OUT.glob(f"en_{group}_*.txt"))) + \
           len(list(OUT.glob(f"ru_{group}_*.txt"))) + \
           len(list(OUT.glob(f"en_{group}_*.txt.gz"))) + \
           len(list(OUT.glob(f"ru_{group}_*.txt.gz")))


def list_titles(group: str, target: int, en_cats: list[str],
                ru_cats: list[str]) -> dict[str, list[str]]:
    """Collect unique titles per language until the joint queue is fat."""
    queue: dict[str, list[str]] = {"en": [], "ru": []}
    seen: set[str] = set()
    for lang, cats in (("en", en_cats), ("ru", ru_cats)):
        for cat in cats:
            needed = target * 2 - sum(len(v) for v in queue.values())
            if needed <= 0:
                return queue
            titles = category_members(lang, cat, depth=1)
            fresh = [t for t in titles if t not in seen]
            seen.update(fresh)
            queue[lang].extend(fresh)
            print(f"  [{group}] listed {cat} -> +{len(fresh)} (queue "
                  f"{len(queue['en'])}en/{len(queue['ru'])}ru)", flush=True)
    return queue


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    grand_new = 0
    for group, (target, en_cats, ru_cats) in PLAN.items():
        have = have_for(group)
        need = target - have
        print(f"=== {group}: target {target}, have {have}, need {max(0, need)}",
              flush=True)
        if need <= 0:
            continue
        queue = list_titles(group, need, en_cats, ru_cats)
        new = 0
        for lang, titles in queue.items():
            if new >= need:
                break
            pending: list[str] = []
            for title in titles:
                out = OUT / f"{lang}_{group}_{slug(title)}.txt"
                if not _have(out):
                    pending.append(title)
            pending = pending[:need - new + 300]
            if not pending:
                continue
            print(f"  [{group}/{lang}] fetching {len(pending)} titles", flush=True)
            for i in range(0, len(pending), 20):
                batch = pending[i:i + 20]
                results = fetch_extracts(lang, batch)
                for title in batch:
                    text = results.get(title)
                    if not text:
                        for got in results:
                            if got.lower() == title.lower():
                                text = results[got]
                                break
                    if not text:
                        continue
                    out = OUT / f"{lang}_{group}_{slug(title)}.txt"
                    if not _have(out):
                        out.write_text(text, encoding="utf-8")
                        new += 1
                if (i // 20) % 5 == 0:
                    print(f"  [{group}/{lang}] {i + len(batch)}/{len(pending)}"
                          f" +{new} new", flush=True)
                time.sleep(max(PAUSE_S, 0.8))
                if new >= need:
                    break
        total_now = have_for(group)
        grand_new += max(0, total_now - have)
        print(f"=== {group} done: was {have}, now {total_now} (+{total_now - have})",
              flush=True)

    files = len(list(OUT.glob("*.txt"))) + len(list(OUT.glob("*.txt.gz")))
    mb = sum(p.stat().st_size for p in OUT.glob("*.txt")) / 1e6
    print(f"TOTAL survival dir: {files} files, {mb:.1f} MB plain text, "
          f"+{grand_new} new this run", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
