# -*- coding: utf-8 -*-
"""Mega-fetch: ~1000 medicine + ~500 electronics + ~500 radio articles.

Strategy: manual lists can't reach 1000 titles, so we walk Wikipedia
CATEGORIES (one level of subcategories deep), collect article titles,
dedupe against what's already on disk, then fetch extracts one by one.

Transport: urllib only (httpx gets 403 from Wikimedia on this box).
Resume-safe: existing non-empty files are skipped. 429-aware: on rate
limit the script backs off exponentially instead of dying.

Run from extra-code root:  python -m data.fetch_mega
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

OUT = Path("data/corpus/survival")
OUT.mkdir(parents=True, exist_ok=True)

HEADERS = {"User-Agent": "extra-code-kb/1.1 (local research; contact: local)"}
PAUSE_S = 0.3          # base pause between API calls
EXTRACT_MIN = 400      # shorter extracts are stubs, skip them

# group -> (target_new_files, en categories, ru categories)
# Categories are walked 1 subcategory deep (subcategory articles count
# toward the same group).
PLAN = {
    "medicine": (
        1000,
        [
            "Category:First aid",
            "Category:Medical emergencies",
            "Category:Symptoms and signs",
            "Category:Diseases and disorders",
            "Category:Infectious diseases",
            "Category:Pharmacology",
            "Category:Drugs",
            "Category:Medical terminology",
            "Category:Surgery",
            "Category:Medical instruments",
        ],
        [
            "Категория:Первая помощь",
            "Категория:Заболевания по алфавиту",
            "Категория:Симптомы по алфавиту",
            "Категория:Фармакология",
            "Категория:Неотложные состояния",
            "Категория:Инфекционные заболевания",
            "Категория:Хирургия",
        ],
    ),
    "electronics": (
        500,
        [
            "Category:Electronics",
            "Category:Electronic components",
            "Category:Electronic engineering",
            "Category:Electronic circuits",
            "Category:Soldering",
            "Category:Electric power",
            "Category:Electricity",
        ],
        [
            "Категория:Электроника",
            "Категория:Радиоэлектроника",
            "Категория:Электронные компоненты",
            "Категория:Электротехника",
        ],
    ),
    "radio": (
        500,
        [
            "Category:Radio technology",
            "Category:Amateur radio",
            "Category:Radio electronics",
            "Category:Radio communication",
            "Category:Radio broadcasting",
            "Category:Antennas (radio)",
            "Category:Wireless",
            "Category:Signal processing",
        ],
        [
            "Категория:Радио",
            "Категория:Радиосвязь",
            "Категория:Радиоэлектроника",
            "Категория:Радиолюбительство",
        ],
    ),
    "chem": (
        500,
        [
            "Category:Chemistry",
            "Category:Chemical elements",
            "Category:Chemical compounds",
            "Category:Chemical reactions",
            "Category:Analytical chemistry",
            "Category:Organic chemistry",
            "Category:Inorganic chemistry",
            "Category:Electrochemistry",
            "Category:Chemical processes",
            "Category:Acids",
            "Category:Chemical synthesis",
        ],
        [
            "Категория:Химия",
            "Категория:Химические элементы",
            "Категория:Химические соединения",
            "Категория:Неорганические соединения",
            "Категория:Органическая химия",
        ],
    ),
    "making": (
        500,
        [
            "Category:Metalworking",
            "Category:Woodworking",
            "Category:Manufacturing",
            "Category:Crafts",
            "Category:Do it yourself",
            "Category:Tools",
            "Category:Materials",
            "Category:Casting",
            "Category:Welding",
            "Category:Sewing",
            "Category:Construction",
        ],
        [
            "Категория:Металлообработка",
            "Категория:Деревообработка",
            "Категория:Ремёсла",
            "Категория:Инструменты",
            "Категория:Самоделки",
        ],
    ),
    "math": (
        500,
        [
            "Category:Mathematics",
            "Category:Mathematical analysis",
            "Category:Algebra",
            "Category:Geometry",
            "Category:Calculus",
            "Category:Probability theory",
            "Category:Statistics",
            "Category:Linear algebra",
            "Category:Number theory",
            "Category:Mathematical theorems",
            "Category:Trigonometry",
            "Category:Discrete mathematics",
        ],
        [
            "Категория:Математика",
            "Категория:Алгебра",
            "Категория:Геометрия",
            "Категория:Математический анализ",
            "Категория:Теория вероятностей",
            "Категория:Теоремы",
            "Категория:Тригонометрия",
        ],
    ),
    "coding": (
        500,
        [
            "Category:Programming languages",
            "Category:Computer programming",
            "Category:Algorithms",
            "Category:Data structures",
            "Category:Software engineering",
            "Category:Software design patterns",
            "Category:Python (programming language)",
            "Category:C++",
            "Category:C programming language",
            "Category:SQL",
            "Category:Version control",
            "Category:Computer security",
        ],
        [
            "Категория:Программирование",
            "Категория:Алгоритмы",
            "Категория:Языки программирования",
            "Категория:Структуры данных",
            "Категория:Программная инженерия",
        ],
    ),
    "rus_lang": (
        100,
        [
            "Category:Russian language",
        ],
        [
            "Категория:Русский язык",
            "Категория:Орфография русского языка",
            "Категория:Пунктуация",
            "Категория:Грамматика русского языка",
            "Категория:Морфология русского языка",
            "Категория:Синтаксис русского языка",
        ],
    ),
}

# Parallel translation pairs: fetch BOTH language versions, save as one
# file with the original text followed by the Russian text. The model
# learns the mapping, retrieval matches either language.
EN_RU_PAIRS = [
    "Physics", "Quantum mechanics", "Theory of relativity", "Thermodynamics",
    "Electromagnetism", "Electricity", "Magnetism", "Optics", "Laser",
    "Nanotechnology", "Internet", "Computer", "Operating system", "Database",
    "Machine learning", "Neural network", "Artificial intelligence",
    "Cryptography", "Encryption", "Blockchain", "Cryptocurrency",
    "HTTP", "TCP/IP", "DNS", "Firewall (computing)", "Cloud computing",
    "Big data", "Data science", "Robot", "Automation", "3D printing",
    "Battery (electricity)", "Lithium-ion battery", "Solar cell",
    "Nuclear power", "Wind turbine", "Hydroelectricity", "Turbine",
    "Internal combustion engine", "Electric motor", "Transmission (mechanics)",
    "Steel", "Concrete", "Glass", "Plastic", "Rubber", "Aluminium",
    "Copper", "Gold", "Petroleum", "Natural gas", "Coal",
    "Evolution", "DNA", "Genetics", "Natural selection", "Cell (biology)",
    "Protein", "Metabolism", "Photosynthesis", "Immune system", "Virus",
    "Bacteria", "Vaccine", "Antibiotic", "Cancer", "Brain",
    "Neuroscience", "Memory", "Sleep", "Nutrition", "Vitamin",
    "Sun", "Moon", "Mars", "Jupiter", "Solar System", "Milky Way",
    "Black hole", "Space exploration", "Rocket", "Satellite",
    "Economics", "Inflation", "Gross domestic product", "Interest rate",
    "Stock market", "Investment", "Supply and demand", "Money",
    "History of writing", "Printing", "Paper", "Library", "Education",
    "Logic", "Philosophy of science", "Statistics",
]

ZH_RU_PAIRS = [
    ("中国", "Китай"), ("长城", "Великая Китайская стена"), ("汉字", "Китайское письмо"),
    ("中医学", "Традиционная китайская медицина"), ("火药", "Порох"),
    ("造纸术", "Бумага"), ("指南针", "Компас"), ("活字印刷术", "Книгопечатание"),
    ("孔子", "Конфуций"), ("老子", "Лао-цзы"), ("孙子兵法", "Искусство войны"),
    ("长江", "Янцзы"), ("黄河", "Хуанхэ"), ("北京", "Пекин"), ("上海", "Шанхай"),
    ("秦始皇", "Цинь Шихуанди"), ("唐朝", "Империя Тан"), ("宋朝", "Империя Сун"),
    ("明朝", "Империя Мин"), ("清朝", "Империя Цин"), ("茶叶", "Чай"),
    ("瓷器", "Фарфор"), ("丝绸", "Шёлк"), ("武术", "Ушу"), ("太极拳", "Тайцзицюань"),
    ("针灸", "Иглоукалывание"), ("高速铁路", "Скоростная железная дорога"),
    ("深圳", "Шэньчжэнь"), ("人工智能", "Искусственный интеллект"),
    ("汉服", "Ханьфу"), ("书法", "Каллиграфия"), ("京剧", "Пекинская опера"),
    ("春节", "Китайский Новый год"), ("中秋节", "Праздник середины осени"),
    ("佛教", "Буддизм"), ("道教", "Даосизм"), ("易经", "Ицзин"),
    ("四书五经", "Четверокнижие"), ("科举", "Кэцзюй"), ("丝绸之路", "Великий шёлковый путь"),
    ("京杭大运河", "Великий канал"), ("故宫", "Запретный город"),
    ("天坛", "Храм Неба"), ("兵马俑", "Терракотовая армия"),
    ("敦煌", "Дуньхуан"), ("禅宗", "Чань-буддизм"), ("功夫", "Кунг-фу"),
    ("茶道", "Чайная церемония"), ("围棋", "Го"), ("象棋", "Сянци"),
    ("火锅", "Хого"), ("豆腐", "Тофу"), ("面条", "Лапша"), ("米饭", "Рис"),
    ("筷子", "Палочки для еды"), ("灯笼", "Китайский фонарик"),
    ("龙", "Дракон"), ("凤凰", "Феникс"), ("玉器", "Нефрит"),
    ("青铜器", "Бронзовые изделия"), ("玉玺", "Императорская печать"),
    ("拼音", "Пиньинь"), ("普通话", "Путунхуа"), ("粤语", "Кантонский диалект"),
    ("方言", "Диалекты китайского языка"), ("成语", "Чэнъюй"),
    ("唐诗", "Танская поэзия"), ("宋词", "Поэзия эпохи Сун"),
    ("红楼梦", "Сон в красном тереме"), ("西游记", "Путешествие на Запад"),
    ("三国演义", "Троецарствие"), ("水浒传", "Речные заводи"),
    ("孔子学院", "Институт Конфуция"), ("春节联欢晚会", "Новогодний гала-концерт CCTV"),
    ("经济改革", "Реформы Дэн Сяопина"), ("深圳经济特区", "Особая экономическая зона Шэньчжэнь"),
    ("中国铁路", "Железные дороги Китая"), ("中国航天", "Космическая программа Китая"),
    ("北斗卫星导航系统", "BeiDou"), ("量子通信", "Квантовая связь"),
    ("超级计算机", "Суперкомпьютер"), ("移动支付", "Мобильные платежи"),
    ("电子商务", "Электронная коммерция"), ("环境保护", "Охрана окружающей среды"),
    ("可再生能源", "Возобновляемая энергетика"), ("中医针灸", "Акупунктура"),
    ("中药学", "Китайская фармакология"), ("气功", "Цигун"),
]


def api_get(lang: str, params: dict[str, str]) -> dict:
    """POST when the query string would blow past URL limits (414)."""
    base = f"https://{lang}.wikipedia.org/w/api.php"
    query = urllib.parse.urlencode(params)
    if len(query) > 1800:
        req = urllib.request.Request(
            base, data=query.encode("utf-8"), headers={
                **HEADERS,
                "Content-Type": "application/x-www-form-urlencoded",
            })
    else:
        req = urllib.request.Request(base + "?" + query, headers=HEADERS)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:  # rate limited: back off, retry
                wait = 5 * (2 ** attempt)
                print(f"  429, backing off {wait}s", flush=True)
                time.sleep(wait)
                continue
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"  net err: {exc}, retry", flush=True)
            time.sleep(3)
    return {}


def category_members(lang: str, cat: str, depth: int = 1) -> list[str]:
    """All article titles in a category (+ subcategories if depth > 0)."""
    titles: list[str] = []
    subcats: list[str] = []
    cont = ""
    while True:
        params = {
            "action": "query", "list": "categorymembers",
            "cmtitle": cat, "cmtype": "page|subcat", "cmlimit": "500",
            "format": "json",
        }
        if cont:
            params["cmcontinue"] = cont
        body = api_get(lang, params)
        for member in body.get("query", {}).get("categorymembers", []):
            title = member.get("title", "")
            if not title or ":" in title.replace("Category:", "", 1).replace("Категория:", "", 1) and (title.startswith("File:") or title.startswith("Wikipedia:") or title.startswith("Help:") or title.startswith("Template:") or title.startswith("Portal:")):
                continue
            if member.get("type") == "subcat":
                subcats.append(title)
            else:
                titles.append(title)
        cont = body.get("continue", {}).get("cmcontinue", "")
        if not cont:
            break
        time.sleep(PAUSE_S)
    if depth > 0:
        for sub in subcats[:15]:  # cap subcategory fan-out
            titles += category_members(lang, sub, depth=depth - 1)
            time.sleep(PAUSE_S)
    return titles


def fetch_extract(lang: str, title: str) -> str | None:
    """Single-title wrapper kept for pairs and one-offs."""
    body = api_get(lang, {
        "action": "query", "prop": "extracts", "explaintext": "1",
        "redirects": "1", "format": "json", "titles": title,
    })
    pages = body.get("query", {}).get("pages", {})
    for page in pages.values():
        text = page.get("extract")
        if text and len(text) > EXTRACT_MIN and "may refer to:" not in text[:300]:
            return text
    return None


def fetch_extracts(lang: str, titles: list[str]) -> dict[str, str]:
    """Batched extracts: up to 20 titles per request (API cap).

    One request per 20 articles instead of 20 requests — this is what
    keeps Wikipedia's rate limiter out of the conversation. Returns
    {title: text} for titles that produced a usable extract.
    """
    out: dict[str, str] = {}
    for i in range(0, len(titles), 20):
        batch = titles[i:i + 20]
        body = api_get(lang, {
            "action": "query", "prop": "extracts", "explaintext": "1",
            "redirects": "1", "format": "json", "exlimit": "20",
            "titles": "|".join(batch),
        })
        # map pageid -> resolved title (redirects rewrite titles)
        pages = body.get("query", {}).get("pages", {})
        for page in pages.values():
            text = page.get("extract")
            title = page.get("title", "")
            if text and len(text) > EXTRACT_MIN and \
                    "may refer to:" not in text[:300]:
                out[title] = text
        time.sleep(1.0)  # per-batch pause: 20 articles per hit
    return out


def _have(out: Path) -> bool:
    """Resume check: article exists plain or already gzipped."""
    if out.exists() and out.stat().st_size > 1000:
        return True
    gz = out.with_suffix(".txt.gz")
    return gz.exists() and gz.stat().st_size > 500


def slug(title: str) -> str:
    s = re.sub(r"[^0-9A-Za-zА-Яа-яЁё]+", "_", title.strip())
    return s.strip("_")[:80]


def main() -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    stats: dict[str, dict[str, int]] = {}
    for group, (target, en_cats, ru_cats) in PLAN.items():
        ok = fail = skipped = 0
        seen: set[str] = set()

        # existing group files count toward the target
        existing = list(OUT.glob(f"en_{group}_*.txt")) + list(OUT.glob(f"ru_{group}_*.txt"))
        have = len(existing)
        print(f"=== {group}: target {target}, already on disk {have}", flush=True)
        if have >= target:
            stats[group] = {"have": have, "fetched": 0, "failed": 0}
            continue

        queue: list[tuple[str, str]] = []  # (lang, title)
        for cat in en_cats:
            for title in category_members("en", cat):
                if title not in seen:
                    seen.add(title)
                    queue.append(("en", title))
            print(f"  listed {cat} -> queue {len(queue)}", flush=True)
            if len(queue) >= target * 3:
                break
        for cat in ru_cats:
            if len(queue) >= target * 3:
                break
            for title in category_members("ru", cat):
                if title not in seen:
                    seen.add(title)
                    queue.append(("ru", title))
            print(f"  listed {cat} -> queue {len(queue)}", flush=True)

        for lang, title in queue:
            if ok + have >= target:
                break
            out = OUT / f"{lang}_{group}_{slug(title)}.txt"
            if _have(out):
                skipped += 1

        # batched fetch: pending titles per language, 20 per request
        pending: dict[str, list[str]] = {"en": [], "ru": []}
        for lang, title in queue:
            out = OUT / f"{lang}_{group}_{slug(title)}.txt"
            if not _have(out):
                pending[lang].append(title)
        for lang, titles in pending.items():
            if not titles:
                continue
            titles = titles[: max(0, target - have - ok) + 200]
            print(f"  batching {lang}: {len(titles)} titles", flush=True)
            results = fetch_extracts(lang, titles)
            for title in titles:
                resolved = results.get(title)
                if not resolved:
                    # maybe saved under a redirected name — try prefix match
                    for got_title, got_text in results.items():
                        if got_title.lower() == title.lower():
                            resolved = got_text
                            break
                out = OUT / f"{lang}_{group}_{slug(title)}.txt"
                if resolved and not _have(out):
                    out.write_text(resolved, encoding="utf-8")
                    ok += 1
                elif not resolved:
                    fail += 1
            have_now = len(list(OUT.glob(f"en_{group}_*.txt"))) + \
                       len(list(OUT.glob(f"ru_{group}_*.txt")))
            print(f"  {group}/{lang}: +{ok} new this run, group total {have_now}",
                  flush=True)
        stats[group] = {"have": have, "fetched": ok, "failed": fail,
                        "skipped": skipped}
        print(f"=== {group} done: had {have}, +{ok} new, fail {fail}", flush=True)

    # --- parallel translation pairs -----------------------------------
    pair_stats = {"en_ru": 0, "zh_ru": 0}
    for en_title in EN_RU_PAIRS:
        out = OUT / f"pair_en_ru_{slug(en_title)}.txt"
        if _have(out):
            pair_stats["en_ru"] += 1
            continue
        en_text = fetch_extract("en", en_title)
        time.sleep(PAUSE_S)
        # ru title: same meaning — let the API resolve the interwiki by
        # searching the en title on ru wiki first, fallback to literal.
        ru_text = fetch_extract("ru", en_title)
        time.sleep(PAUSE_S)
        if en_text and ru_text:
            out.write_text(
                f"[EN] {en_title}\n\n{en_text}\n\n=== ПЕРЕВОД (RU) ===\n\n{ru_text}",
                encoding="utf-8")
            pair_stats["en_ru"] += 1
    print(f"=== en_ru pairs: {pair_stats['en_ru']}/{len(EN_RU_PAIRS)}", flush=True)

    for zh_title, ru_title in ZH_RU_PAIRS:
        out = OUT / f"pair_zh_ru_{slug(ru_title)}.txt"
        if _have(out):
            pair_stats["zh_ru"] += 1
            continue
        zh_text = fetch_extract("zh", zh_title)
        time.sleep(PAUSE_S)
        ru_text = fetch_extract("ru", ru_title)
        time.sleep(PAUSE_S)
        if zh_text and ru_text:
            out.write_text(
                f"[ZH] {zh_title}\n\n{zh_text}\n\n=== ПЕРЕВОД (RU) ===\n\n{ru_text}",
                encoding="utf-8")
            pair_stats["zh_ru"] += 1
    print(f"=== zh_ru pairs: {pair_stats['zh_ru']}/{len(ZH_RU_PAIRS)}", flush=True)

    total = sum(1 for _ in OUT.glob("*.txt"))
    mb = sum(p.stat().st_size for p in OUT.glob("*.txt")) / 1e6
    print(json.dumps({"stats": stats, "pairs": pair_stats,
                      "files_total": total,
                      "mb_total": round(mb, 1)}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
