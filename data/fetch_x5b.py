# -*- coding: utf-8 -*-
"""Second wave of the x5 expansion: pharmacy, medicinal plants, energy,
workshop skills, household chemistry, toxic flora/fauna.

Complements fetch_x5 (history/medicine/radio/electronics/cooking/
survival/languages/coding). Wrong category names simply return empty
and are skipped; the run is resume-safe, so rerunning after fixing a
name picks up only the missing files.

Run from extra-code root:  python -m data.fetch_x5b
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from data.fetch_mega import (
    OUT, PAUSE_S, _have, category_members, fetch_extracts, slug,
)

PLAN = {
    "pharm": (
        1500,
        [
            "Category:Pharmacology", "Category:Pharmaceutical drugs",
            "Category:Antidotes", "Category:Anesthetics",
            "Category:Analgesics", "Category:Antipyretics",
            "Category:Antiseptics", "Category:Disinfectants",
            "Category:Antiviral drugs", "Category:Antifungal drugs",
            "Category:Antiparasitic agents", "Category:Sedatives",
            "Category:Vitamins", "Category:Insulin",
            "Category:Adrenergic drugs", "Category:Antibiotics",
        ],
        [
            "Категория:Фармакология", "Категория:Лекарственные средства",
            "Категория:Обезболивающие средства", "Категория:Жаропонижающие средства",
            "Категория:Антисептики", "Категория:Дезинфицирующие средства",
            "Категория:Противовирусные препараты", "Категория:Противогрибковые средства",
            "Категория:Витамины", "Категория:Антидоты",
            "Категория:Снотворные средства", "Категория:Антибиотики",
        ],
    ),
    "natmed": (
        1200,
        [
            "Category:Medicinal plants", "Category:Herbalism",
            "Category:Traditional medicine", "Category:Essential oils",
            "Category:Phytotherapy", "Category:Home remedies",
            "Category:Naturopathy",
        ],
        [
            "Категория:Лекарственные растения", "Категория:Народная медицина",
            "Категория:Фитотерапия", "Категория:Эфирные масла",
            "Категория:Растительные яды",
        ],
    ),
    "energy": (
        1200,
        [
            "Category:Electric batteries", "Category:Rechargeable batteries",
            "Category:Battery chargers", "Category:Solar energy",
            "Category:Photovoltaics", "Category:Wind power",
            "Category:Generators", "Category:Uninterruptible power supplies",
            "Category:Power inverters", "Category:Electric power distribution",
            "Category:Transformers", "Category:Energy storage",
            "Category:Biofuels", "Category:Heat pumps",
        ],
        [
            "Категория:Электрические аккумуляторы", "Категория:Солнечная энергетика",
            "Категория:Ветроэнергетика", "Категория:Генераторы",
            "Категория:Трансформаторы", "Категория:Электроснабжение",
            "Категория:Биотопливо", "Категория:Тепловые насосы",
        ],
    ),
    "workshop": (
        1500,
        [
            "Category:Metalworking", "Category:Woodworking",
            "Category:Welding", "Category:Machining",
            "Category:Casting", "Category:Blacksmithing",
            "Category:Soldering", "Category:Hand tools",
            "Category:Power tools", "Category:Measuring instruments",
            "Category:Plumbing", "Category:Fasteners",
            "Category:Concrete",
            "Category:Roofing", "Category:Electrical wiring",
            "Category:Locks (hardware)", "Category:Sharpening",
        ],
        [
            "Категория:Металлообработка", "Категория:Деревообработка",
            "Категория:Сварка", "Категория:Пайка",
            "Категория:Инструменты", "Категория:Крепёжные изделия",
            "Категория:Слесарное дело", "Категория:Столярное дело",
            "Категория:Электромонтаж", "Категория:Замки",
            "Категория:Заточка", "Категория:Кузнечное дело",
        ],
    ),
    "chemhome": (
        1000,
        [
            "Category:Cleaning products", "Category:Soaps",
            "Category:Detergents", "Category:Bleaches",
            "Category:Adhesives", "Category:Paints",
            "Category:Coatings", "Category:Sealants",
            "Category:Waxes", "Category:Solvents",
            "Category:Household chemicals", "Category:Water softening",
        ],
        [
            "Категория:Бытовая химия", "Категория:Мыло",
            "Категория:Моющие средства", "Категория:Отбеливатели",
            "Категория:Клеи", "Категория:Краски",
            "Категория:Растворители", "Категория:Воски",
        ],
    ),
    "toxic": (
        800,
        [
            "Category:Poisonous plants", "Category:Toxic fungi",
            "Category:Venomous animals", "Category:Poisonous animals",
            "Category:Mushroom poisoning", "Category:Plant poisoning",
            "Category:Foodborne illnesses", "Category:Mycotoxins",
        ],
        [
            "Категория:Ядовитые растения", "Категория:Ядовитые грибы",
            "Категория:Несъедобные грибы", "Категория:Ядовитые животные",
            "Категория:Пищевые отравления", "Категория:Микотоксины",
            "Категория:Отравления",
        ],
    ),
}


def have_for(group: str) -> int:
    return sum(
        len(list(OUT.glob(f"{lang}_{group}_*{ext}")))
        for lang in ("en", "ru")
        for ext in (".txt", ".txt.gz")
    )


def list_titles(group: str, needed: int, en_cats: list[str],
                ru_cats: list[str]) -> dict[str, list[str]]:
    queue: dict[str, list[str]] = {"en": [], "ru": []}
    seen: set[str] = set()
    for lang, cats in (("en", en_cats), ("ru", ru_cats)):
        for cat in cats:
            if sum(len(v) for v in queue.values()) >= needed * 2:
                return queue
            titles = category_members(lang, cat, depth=1)
            fresh = [t for t in titles if t not in seen]
            seen.update(fresh)
            queue[lang].extend(fresh)
            print(f"  [{group}] listed {cat} -> +{len(fresh)} "
                  f"(queue {len(queue['en'])}en/{len(queue['ru'])}ru)", flush=True)
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
                    print(f"  [{group}/{lang}] {i + len(batch)}/{len(pending)} "
                          f"+{new} new", flush=True)
                time.sleep(max(PAUSE_S, 0.8))
                if new >= need:
                    break
        total_now = have_for(group)
        grand_new += max(0, total_now - have)
        print(f"=== {group} done: was {have}, now {total_now} (+{total_now - have})",
              flush=True)

    print(f"second wave done, +{grand_new} files this run", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
