# -*- coding: utf-8 -*-
"""Wave 5: deep 2025-2026 facts + core-article deepening per section.

Part A: monthly current-events portals (EN), 2025-01 .. 2026-12 — dense
        day-by-day facts.
Part B: topic year pages (EN): aviation, spaceflight, video games,
        music, film, television, sports, paleontology, archaeology,
        medicine, nobel prizes x6, AI timeline, computing timeline.
Part C: RU mirrors where they exist (try, skip 404).
Part D: one gold core article per knowledge section (EN+RU): electronics,
        radio, medicine/pharma, energy, cooking, toxicity, first aid.

Output: data/corpus/facts_2026/ (year material) and
        data/corpus/deepen_2026/ (core articles). kb.py indexes both
        automatically.
"""
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(r"C:\Users\Sem52\Documents\Default Project\extra-code")
FACTS = ROOT / "data" / "corpus" / "facts_2026"
DEEP = ROOT / "data" / "corpus" / "deepen_2026"
FACTS.mkdir(parents=True, exist_ok=True)
DEEP.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "ExtraCodeKB/1.0 (personal local knowledge base; no mirroring)"}

MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]

PORTALS = [f"Portal:Current_events/{m}_{y}" for y in (2025, 2026) for m in MONTHS]

TOPIC_YEARS = [
    "2025_in_aviation", "2026_in_aviation",
    "2025_in_spaceflight", "2026_in_spaceflight",
    "2025_in_video_games", "2026_in_video_games",
    "2025_in_music", "2026_in_music",
    "2025_in_film", "2026_in_film",
    "2025_in_television", "2026_in_television",
    "2025_in_sports", "2026_in_sports",
    "2025_in_paleontology", "2025_in_archaeology",
    "2026_in_paleontology", "2026_in_archaeology",
    "2025_in_medicine", "2026_in_medicine",
    "2025_in_international_relations",
    "2025_Nobel_Prize_in_Physics", "2025_Nobel_Prize_in_Chemistry",
    "2025_Nobel_Prize_in_Physiology_or_Medicine",
    "2025_Nobel_Peace_Prize", "2025_Nobel_Memorial_Prize_in_Economic_Sciences",
    "2025_Nobel_Prize_in_Literature",
    "Timeline_of_artificial_intelligence",
    "Timeline_of_computing",
    "2025_United_Nations_Climate_Change_Conference",
]

RU_TRIES = [
    "Портал:Текущие_события/Январь_2025", "Портал:Текущие_события/Февраль_2025",
    "2025_год_в_кино", "2025_год_в_музыке", "2026_год_в_науке",
]

CORE_EN = [
    # electronics
    "Electronics", "Operational_amplifier", "Bipolar_junction_transistor",
    "MOSFET", "Printed_circuit_board", "Integrated_circuit",
    "Analog-to-digital_converter", "Oscilloscope",
    # radio
    "Radio", "Shortwave_radio", "Amateur_radio", "Radio_propagation",
    "Antenna_(radio)", "Software-defined_radio", "Frequency_modulation",
    # medicine / pharmacy
    "Medicine", "Pharmacology", "Antibiotic", "Analgesic",
    "First_aid", "Cardiopulmonary_resuscitation",
    # energy
    "Renewable_energy", "Solar_power", "Nuclear_power",
    "Electric_battery", "Lithium-ion_battery", "Wind_power",
    # cooking / food
    "Cooking", "Food_safety", "Foodborne_illness", "Nutrition",
    # toxicity / survival
    "Toxicity", "Poison", "Wilderness_emergency_care", "Firemaking",
]

CORE_RU = [
    "Электроника", "Радио", "Радиосвязь", "Фармакология", "Первая_помощь",
    "Возобновляемая_энергетика", "Солнечная_энергетика", "Приготовление_пищи",
    "Токсичность", "Выживание",
]


def fetch(url: str, timeout: int = 90) -> bytes | None:
    for attempt in range(1, 6):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except Exception as exc:
            code = getattr(exc, "code", None)
            if code == 404:
                return None
            wait = 30 if code == 429 else 2 * attempt
            print(f"  fail ({attempt}/5) {url.rsplit('/', 1)[-1][:44]}: "
                  f"{exc} -> wait {wait}s", flush=True)
            time.sleep(wait)
    return None


def strip_wikitext(wt: str) -> str:
    wt = re.sub(r"<ref[^>]*/>", "", wt)
    wt = re.sub(r"<ref[^>]*>.*?</ref>", "", wt, flags=re.S)
    for _ in range(6):
        wt = re.sub(r"\{\{[^{}]*\}\}", "", wt)
    wt = re.sub(r"\{\|.*?\|\}", "", wt, flags=re.S)
    wt = re.sub(r"<![^>]*-->", "", wt)
    wt = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", wt)
    wt = re.sub(r"'''?", "", wt)
    wt = re.sub(r"<[^>]+>", "", wt)
    wt = re.sub(r"\n{3,}", "\n\n", wt)
    return wt.strip()


def pull(lang: str, page: str, out_dir: Path, tag: str) -> bool:
    out = out_dir / f"{tag}.md"
    if out.exists():
        return True
    url = (f"https://{lang}.wikipedia.org/w/index.php?title="
           + urllib.parse.quote(page.replace(" ", "_"), safe=":/")
           + "&action=raw")
    data = fetch(url)
    time.sleep(1.2)  # wikipedia rate limit
    if not data:
        return False
    text = strip_wikitext(data.decode("utf-8", errors="ignore"))
    if len(text) < 400:
        return False
    out.write_text(text, encoding="utf-8")
    print(f"[{lang}] {page}: {len(text)//1024} KB", flush=True)
    return True


def slug(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9а-яА-Я]+", "_", s)
    return s.strip("_").lower()


def main() -> int:
    t0 = time.time()
    ok = miss = 0
    for p in PORTALS:
        if pull("en", p, FACTS, f"cur_{slug(p)}"):
            ok += 1
        else:
            miss += 1
    for p in TOPIC_YEARS:
        if pull("en", p, FACTS, f"yr_{slug(p)}"):
            ok += 1
        else:
            miss += 1
    for p in RU_TRIES:
        if pull("ru", p, FACTS, f"ru_{slug(p)}"):
            ok += 1
        else:
            miss += 1
    for p in CORE_EN:
        if pull("en", p, DEEP, f"core_{slug(p)}"):
            ok += 1
        else:
            miss += 1
    for p in CORE_RU:
        if pull("ru", p, DEEP, f"core_ru_{slug(p)}"):
            ok += 1
        else:
            miss += 1
    print(f"DONE ok={ok} miss={miss} elapsed={int(time.time()-t0)}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
