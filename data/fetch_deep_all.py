# -*- coding: utf-8 -*-
"""Wave 6: languages, math, physics, C++ and Python — everything maxed.

  langs   : grammar + language overview pages (EN+RU)
  math    : full core math structure (EN+RU)
  physics : full core physics structure (EN+RU)
  cpp     : modern-cpp-tutorial, modern-cpp-features, cpp_best_practices
  python  : FULL stdlib via GitHub API, python-patterns repo, key PEPs

Output dirs: langs_2026/, math_2026/, physics_2026/ (wiki),
docs_2026/ (cpp/py docs), real_code_py/ (patterns source).
kb.py auto-indexes all of them.
"""
import io
import json
import re
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(r"C:\Users\Sem52\Documents\Default Project\extra-code")
BASE = ROOT / "data" / "corpus"
LANGS = BASE / "langs_2026"
MATH = BASE / "math_2026"
PHYS = BASE / "physics_2026"
DOCS = BASE / "docs_2026"
PYCODE = BASE / "real_code_py"
for d in (LANGS, MATH, PHYS, DOCS, PYCODE):
    d.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "ExtraCodeKB/1.0 (personal local knowledge base; no mirroring)"}

# ---------------------------------------------------------------- wiki
WIKI_SECTIONS = {
    LANGS: [
        ("en", "English_grammar"), ("en", "Russian_grammar"),
        ("en", "Chinese_grammar"), ("en", "Mandarin_Chinese"),
        ("en", "English_language"), ("en", "Russian_language"),
        ("en", "Spanish_language"), ("en", "German_language"),
        ("en", "French_language"), ("en", "Japanese_language"),
        ("en", "Pinyin"), ("en", "Cyrillic_script"),
        ("en", "Chinese_characters"), ("en", "Stroke_order"),
        ("en", "Language_education"), ("en", "Second_language"),
        ("ru", "Русский_язык"), ("ru", "Английский_язык"),
        ("ru", "Китайский_язык"), ("ru", "Немецкий_язык"),
        ("ru", "Французский_язык"), ("ru", "Испанский_язык"),
        ("ru", "Японский_язык"), ("ru", "Грамматика_русского_языка"),
    ],
    MATH: [
        ("en", "Mathematics"), ("en", "Algebra"), ("en", "Linear_algebra"),
        ("en", "Calculus"), ("en", "Mathematical_analysis"),
        ("en", "Probability_theory"), ("en", "Statistics"),
        ("en", "Discrete_mathematics"), ("en", "Number_theory"),
        ("en", "Geometry"), ("en", "Trigonometry"),
        ("en", "Mathematical_logic"), ("en", "Differential_equations"),
        ("en", "Complex_analysis"), ("en", "Topology"),
        ("en", "Group_theory"), ("en", "Combinatorics"),
        ("en", "Mathematical_optimization"), ("en", "Numerical_analysis"),
        ("en", "Mathematical_proof"), ("en", "Set_theory"),
        ("ru", "Математика"), ("ru", "Алгебра"),
        ("ru", "Линейная_алгебра"), ("ru", "Математический_анализ"),
        ("ru", "Теория_вероятностей"), ("ru", "Дискретная_математика"),
        ("ru", "Теория_чисел"), ("ru", "Геометрия"),
        ("ru", "Тригонометрия"), ("ru", "Математическая_логика"),
        ("ru", "Дифференциальные_уравнения"), ("ru", "Комплексный_анализ"),
        ("ru", "Теория_групп"), ("ru", "Комбинаторика"),
    ],
    PHYS: [
        ("en", "Physics"), ("en", "Classical_mechanics"),
        ("en", "Quantum_mechanics"), ("en", "Electromagnetism"),
        ("en", "Thermodynamics"), ("en", "Special_relativity"),
        ("en", "General_relativity"), ("en", "Optics"),
        ("en", "Atomic_physics"), ("en", "Nuclear_physics"),
        ("en", "Particle_physics"), ("en", "Solid-state_physics"),
        ("en", "Condensed_matter_physics"), ("en", "Fluid_dynamics"),
        ("en", "Astrophysics"), ("en", "Statistical_mechanics"),
        ("en", "Wave"), ("en", "Chaos_theory"),
        ("ru", "Физика"), ("ru", "Классическая_механика"),
        ("ru", "Квантовая_механика"), ("ru", "Электромагнетизм"),
        ("ru", "Термодинамика"), ("ru", "Теория_относительности"),
        ("ru", "Оптика"), ("ru", "Ядерная_физика"),
        ("ru", "Физика_твёрдого_тела"), ("ru", "Гидродинамика"),
        ("ru", "Астрофизика"), ("ru", "Статистическая_механика"),
    ],
}


def fetch(url: str, timeout: int = 120) -> bytes | None:
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


def slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9а-яА-ЯёЁ]+", "_", s).strip("_").lower()


def do_wiki() -> None:
    for out_dir, pages in WIKI_SECTIONS.items():
        for lang, page in pages:
            out = out_dir / f"{slug(lang + '_' + page)}.md"
            if out.exists():
                continue
            url = (f"https://{lang}.wikipedia.org/w/index.php?title="
                   + urllib.parse.quote(page.replace(" ", "_")) + "&action=raw")
            data = fetch(url)
            if not data:
                continue
            text = strip_wikitext(data.decode("utf-8", errors="ignore"))
            if len(text) < 400:
                continue
            out.write_text(text, encoding="utf-8")
            print(f"[{lang}] {page}: {len(text)//1024} KB", flush=True)
            time.sleep(1.2)  # wikipedia rate limit: stay under ~1 req/s


# ---------------------------------------------------------------- repos
def repo_zip(repo: str, timeout: int = 400) -> zipfile.ZipFile | None:
    for branch in ("master", "main"):
        data = fetch(f"https://codeload.github.com/{repo}/zip/refs/heads/{branch}",
                     timeout)
        if data:
            try:
                return zipfile.ZipFile(io.BytesIO(data))
            except zipfile.BadZipFile:
                pass
    print(f"[repo] {repo} ZIP FAILED", flush=True)
    return None


def save_repo_md(repo: str, prefix: str, subdir: str = "") -> int:
    zf = repo_zip(repo)
    if zf is None:
        return 0
    saved = 0
    for n in zf.namelist():
        if not n.endswith(".md"):
            continue
        low = n.lower()
        if subdir and subdir not in low:
            continue
        if ".github" in low or "zh" in low.split("/") or "cn" in low.split("/"):
            continue
        try:
            text = zf.read(n).decode("utf-8", errors="ignore")
        except Exception:
            continue
        if len(text.strip()) < 300:
            continue
        stem = slug(Path(n).stem) or "doc"
        out = DOCS / f"{prefix}_{stem}.md"
        if out.exists():
            continue
        out.write_text(text, encoding="utf-8")
        saved += 1
    print(f"[repo] {repo}: saved {saved} md", flush=True)
    return saved


def do_cpp() -> None:
    save_repo_md("changkun/modern-cpp-tutorial", "mcpp")
    save_repo_md("AnthonyCalandra/modern-cpp-features", "cppfeat")
    save_repo_md("cpp-best-practices/cpp_best_practices", "cppbp")
    # the single most valuable C++ document in existence
    out = DOCS / "cpp_core_guidelines.md"
    if not out.exists():
        data = fetch(
            "https://raw.githubusercontent.com/isocpp/CppCoreGuidelines/main/CppCoreGuidelines.md",
            timeout=300,
        )
        if data:
            out.write_text(data.decode("utf-8", errors="ignore"),
                           encoding="utf-8")
            print(f"[cpp] core guidelines: {len(data)//1024} KB", flush=True)


def do_py_patterns() -> None:
    zf = repo_zip("faif/python-patterns")
    if zf is None:
        return
    saved = 0
    for n in zf.namelist():
        if not n.endswith(".py"):
            continue
        info = zf.getinfo(n)
        if info.file_size > 120_000:
            continue
        try:
            text = zf.read(n).decode("utf-8", errors="ignore")
        except Exception:
            continue
        if len(text.strip()) < 200:
            continue
        saved += 1
        (PYCODE / f"pypat_{saved:04d}.txt").write_text(text, encoding="utf-8")
    print(f"[repo] python-patterns: saved {saved} py", flush=True)


def do_py_stdlib() -> None:
    if list(DOCS.glob("cpystd_*.md")):
        print("[stdlib] already present", flush=True)
        return
    listing = fetch(
        "https://api.github.com/repos/python/cpython/contents/Doc/library",
        timeout=90,
    )
    if not listing:
        print("[stdlib] listing FAILED", flush=True)
        return
    try:
        files = json.loads(listing.decode("utf-8"))
    except json.JSONDecodeError:
        print("[stdlib] listing not json", flush=True)
        return
    names = [f["name"] for f in files if f["name"].endswith(".rst")]
    print(f"[stdlib] {len(names)} modules", flush=True)
    saved = 0
    for name in names:
        stem = slug(name[:-4])
        data = fetch(
            f"https://raw.githubusercontent.com/python/cpython/main/Doc/library/{urllib.parse.quote(name)}",
            timeout=60,
        )
        if not data:
            continue
        text = data.decode("utf-8", errors="ignore")
        if len(text.strip()) < 300:
            continue
        (DOCS / f"cpystd_{stem}.md").write_text(text, encoding="utf-8")
        saved += 1
    print(f"[stdlib] saved {saved}/{len(names)}", flush=True)


PEPS = [8, 20, 257, 484, 526, 557, 572, 498, 492, 525, 604, 634, 636, 695, 703, 744]


def do_peps() -> None:
    for num in PEPS:
        out = DOCS / f"pep_{num:04d}.md"
        if out.exists():
            continue
        data = fetch(
            f"https://raw.githubusercontent.com/python/peps/main/peps/pep-{num:04d}.rst",
            timeout=60,
        )
        if not data:
            continue
        text = data.decode("utf-8", errors="ignore")
        if len(text.strip()) < 300:
            continue
        out.write_text(text, encoding="utf-8")
        print(f"[pep] pep-{num:04d}: {len(text)//1024} KB", flush=True)


def main() -> None:
    t0 = time.time()
    do_wiki()
    do_cpp()
    do_py_patterns()
    do_py_stdlib()
    do_peps()
    print(f"DONE elapsed={int(time.time()-t0)}s", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
