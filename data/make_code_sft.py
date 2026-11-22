"""Build the code-teaching SFT set: Python, C++, Java, C, C#, x86-64 asm, RE.

The corpus teaches the model what code looks like; this script teaches it how
to answer code questions. Every row is a real question with a working answer,
not a template with a blank in it, so the model learns the shape of a useful
reply (short explanation -> code -> one gotcha).

Outputs
    data/sft/code_seed.jsonl    finished rows, ready for train/sft.py
    data/sft/code_prompts.jsonl system+user only, fed to data/distill.py

    python -m data.make_code_sft --out data/sft
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SYSTEM = (
    "Ты Extra code — компактная локальная модель для программирования. "
    "Отвечай по делу: короткое объяснение, рабочий код, одна важная деталь."
)

Lang = str


def E(lang: Lang, topic: str, question: str, answer: str, *, level: str = "middle") -> dict:
    return {
        "lang": lang,
        "topic": topic,
        "question": question,
        "answer": answer.strip(),
        "level": level,
    }


TOPICS: list[dict] = []


def add(*entries: dict) -> None:
    TOPICS.extend(entries)


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------
add(
    E("python", "list comprehension vs generator",
      "Чем отличается list comprehension от генератора в Python?",
      """
Список строит все элементы сразу в памяти, генератор выдаёт по одному через `yield` и хранит только состояние.

```python
squares = [x * x for x in range(10)]        # list, 10 ints in memory
squares_gen = (x * x for x in range(10))    # generator, lazy

print(next(squares_gen))  # 0
print(sum(squares_gen))   # 285, the 0 is already consumed
```

Бери генератор, когда данных много или они бесконечные: `(line.strip() for line in fh)` не читает файл целиком. Список нужен, если по нему надо ходить несколько раз или знать длину.
"""),
    E("python", "decorator",
      "Напиши декоратор на Python, который логирует время работы функции.",
      """
```python
import functools
import time

def timed(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed = (time.perf_counter() - start) * 1000
            print(f"{func.__name__} took {elapsed:.2f} ms")
    return wrapper

@timed
def slow_sum(n: int) -> int:
    return sum(range(n))

slow_sum(10**6)
```

`functools.wraps` переносит имя и докстринг, иначе в трейсбеке будет `wrapper`. `finally` гарантирует замер и при исключении.
"""),
    E("python", "context manager",
      "Как сделать свой context manager в Python?",
      """
Два способа: класс с `__enter__`/`__exit__` или `contextlib.contextmanager`.

```python
from contextlib import contextmanager

@contextmanager
def opened(path: str, mode: str = "r"):
    fh = open(path, mode, encoding="utf-8")
    try:
        yield fh
    finally:
        fh.close()

with opened("data.txt") as fh:
    for line in fh:
        print(line.rstrip())
```

`__exit__` получает `(exc_type, exc, tb)`; вернуть `True` — значит погасить исключение. Возвращать `True` почти всегда ошибка: ошибку нужно видеть.
"""),
    E("python", "dataclass",
      "Что даёт @dataclass в Python и когда он не подходит?",
      """
`@dataclass` генерирует `__init__`, `__repr__`, `__eq__` по аннотациям полей.

```python
from dataclasses import dataclass, field

@dataclass(frozen=True, slots=True)
class Point:
    x: float
    y: float
    tags: tuple[str, ...] = ()

p = Point(1.0, 2.0)
print(p)            # Point(x=1.0, y=2.0, tags=())
print(p == Point(1.0, 2.0))
```

`frozen=True` делает объект хэшируемым и неизменяемым, `slots=True` убирает `__dict__` и экономит память. Не подходит, когда нужна сложная валидация в конструкторе — тогда обычный класс или `pydantic`.
"""),
    E("python", "typing protocol",
      "Как в Python описать интерфейс без наследования?",
      """
Через `typing.Protocol` — структурная типизация: подходит любой класс с нужными методами.

```python
from typing import Protocol

class Writer(Protocol):
    def write(self, data: str) -> int: ...

def save(w: Writer, text: str) -> None:
    w.write(text)

class FileWriter:
    def write(self, data: str) -> int:
        return len(data)

save(FileWriter(), "hello")  # ok, no inheritance
```

`Protocol` проверяется mypy/pyright, в рантайме наследование не нужно. Для runtime-проверок есть `@runtime_checkable`, но она смотрит только на наличие имён, не на сигнатуры.
"""),
    E("python", "asyncio gather",
      "Как параллельно выполнить несколько HTTP-запросов в asyncio?",
      """
`asyncio.gather` запускает корутины конкурентно; семафор ограничивает давление на сервер.

```python
import asyncio
import httpx

async def fetch(client: httpx.AsyncClient, url: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        r = await client.get(url, timeout=10)
        r.raise_for_status()
        return r.text[:100]

async def main(urls: list[str]) -> list[str]:
    sem = asyncio.Semaphore(8)
    async with httpx.AsyncClient() as client:
        tasks = [fetch(client, u, sem) for u in urls]
        return await asyncio.gather(*tasks)

asyncio.run(main(["https://example.com"] * 20))
```

`gather` по умолчанию падает на первой ошибке; `return_exceptions=True` возвращает исключения как значения. Один `AsyncClient` на все запросы переиспользует соединения.
"""),
    E("python", "pathlib",
      "Как читать и писать файлы через pathlib?",
      """
```python
from pathlib import Path

p = Path("out") / "notes.txt"
p.parent.mkdir(parents=True, exist_ok=True)

p.write_text("привет\\n", encoding="utf-8")
text = p.read_text(encoding="utf-8")

for path in Path(".").glob("**/*.py"):
    if path.is_file():
        print(path, path.stat().st_size)
```

Всегда указывай `encoding="utf-8"`: на Windows дефолт cp1251 и кириллица ломается. `read_text` читает файл целиком — для больших файлов `open()` и итерация по строкам.
"""),
    E("python", "argparse",
      "Сделай CLI на argparse с обязательным аргументом и флагом.",
      """
```python
import argparse

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tool", description="count lines")
    p.add_argument("path", help="file to read")
    p.add_argument("-n", "--limit", type=int, default=10, help="max lines")
    p.add_argument("-v", "--verbose", action="store_true")
    return p

def main() -> int:
    args = build_parser().parse_args()
    with open(args.path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i >= args.limit:
                break
            if args.verbose:
                print(f"{i:4d} {line.rstrip()}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
```

`parse_args` сам печатает usage и выходит с кодом 2 при ошибке. Возвращай `int` из `main` и отдавай в `SystemExit` — так работают коды возврата в CI.
"""),
    E("python", "gil multiprocessing",
      "Что такое GIL и когда нужен multiprocessing?",
      """
GIL — глобальная блокировка интерпретатора CPython: в один момент байткод исполняет один поток. Потоки помогают на I/O, но не на CPU-вычислениях.

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def cpu_heavy(n: int) -> int:
    return sum(i * i for i in range(n))

if __name__ == "__main__":
    with ThreadPoolExecutor(8) as pool:      # I/O-bound
        list(pool.map(lambda _: None, range(8)))

    with ProcessPoolExecutor() as pool:      # CPU-bound
        print(list(pool.map(cpu_heavy, [10**6] * 4)))
```

Процессы обходят GIL, но платят за сериализацию аргументов и памяти. На Windows процесс стартует заново, поэтому `if __name__ == "__main__"` обязателен.
"""),
    E("python", "pytest",
      "Как написать параметризованный тест на pytest?",
      """
```python
import pytest

def parse_age(raw: str) -> int:
    value = int(raw)
    if not 0 <= value <= 150:
        raise ValueError("age out of range")
    return value

@pytest.mark.parametrize(
    ("raw", "expected"),
    [("0", 0), ("42", 42), ("150", 150)],
)
def test_parse_age_ok(raw: str, expected: int) -> None:
    assert parse_age(raw) == expected

@pytest.mark.parametrize("raw", ["-1", "151", "abc"])
def test_parse_age_bad(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_age(raw)
```

Фикстуры (`@pytest.fixture`) дают общие ресурсы, `tmp_path` — временную папку. `pytest -q` и `pytest -k age` для быстрого прогона.
"""),
    E("python", "mutable default",
      "Почему список в аргументе по умолчанию — плохая идея?",
      """
Значение по умолчанию создаётся один раз при определении функции и переживает вызовы.

```python
def bad(item, bucket=[]):      # bug
    bucket.append(item)
    return bucket

def good(item, bucket=None):
    if bucket is None:
        bucket = []
    bucket.append(item)
    return bucket

print(bad(1), bad(2))    # [1, 2] [1, 2]
print(good(1), good(2))  # [1] [2]
```

Та же ловушка с `{}` и `set()`. Для неизменяемых значений (`None`, числа, строки, кортежи) дефолт безопасен.
"""),
    E("python", "generator pipeline",
      "Как собрать конвейер обработки строк на генераторах?",
      """
```python
from pathlib import Path
from typing import Iterable, Iterator

def read_lines(path: Path) -> Iterator[str]:
    with path.open(encoding="utf-8") as fh:
        yield from fh

def non_empty(lines: Iterable[str]) -> Iterator[str]:
    for line in lines:
        if line.strip():
            yield line.rstrip()

def numbers(lines: Iterable[str]) -> Iterator[int]:
    for line in lines:
        try:
            yield int(line)
        except ValueError:
            continue

total = sum(numbers(non_empty(read_lines(Path("values.txt")))))
print(total)
```

`yield from` пробрасывает значения и исключения вложенного генератора. Конвейер ленивый: файл читается по мере надобности, память не растёт.
"""),
    E("python", "exceptions",
      "Как правильно бросать исключения с сохранением причины?",
      """
```python
class ConfigError(Exception):
    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"bad config {path}: {reason}")
        self.path = path
        self.reason = reason

def load(path: str) -> dict:
    try:
        import json
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as exc:
        raise ConfigError(path, str(exc)) from exc
```

`raise ... from exc` сохраняет цепочку: в трейсбеке видно и исходную ошибку, и обёртку. Не глотай исключения через `except Exception: pass`.
"""),
    E("python", "slots",
      "Зачем нужен __slots__ и как он сочетается с dataclass?",
      """
`__slots__` убирает per-instance `__dict__`: меньше памяти и быстрее доступ к полям, но нельзя добавлять новые атрибуты.

```python
import sys

class WithDict:
    def __init__(self, x: int) -> None:
        self.x = x

class WithSlots:
    __slots__ = ("x",)
    def __init__(self, x: int) -> None:
        self.x = x

print(sys.getsizeof(WithDict(1)))   # ~48 bytes, plus __dict__
print(sys.getsizeof(WithSlots(1)))  # ~32 bytes, no __dict__

from dataclasses import dataclass

@dataclass(slots=True)
class Point:
    x: int
    y: int
```

Для десятков тысяч объектов экономия заметна; для десятка — нет. `slots=True` в dataclass появился в Python 3.10.
"""),
    E("python", "json",
      "Как безопасно прочитать JSON и что делать с битым файлом?",
      """
```python
import json
from pathlib import Path

def load_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: line {exc.lineno}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected an object, got {type(data).__name__}")
    return data

cfg = load_json(Path("config.json"))
```

`json.loads` бросает `JSONDecodeError`, а не возвращает `None`. Проверяй тип после загрузки: `[]` тоже валидный JSON и дальше упадёт на `.get`.
"""),
    E("python", "sorting keys",
      "Как отсортировать список словарей по двум полям?",
      """
```python
rows = [
    {"name": "b", "score": 90},
    {"name": "a", "score": 90},
    {"name": "c", "score": 95},
]

by_name = sorted(rows, key=lambda r: r["name"])
by_score_then_name = sorted(rows, key=lambda r: (-r["score"], r["name"]))
print(by_score_then_name)
# [{'name': 'c', 'score': 95}, {'name': 'a', 'score': 90}, {'name': 'b', 'score': 90}]
```

`sorted` стабильный: равные элементы сохраняют исходный порядок. Минус перед числом даёт сортировку по убыванию, строку оставляй как есть.
"""),
    E("python", "dataclass vs pydantic",
      "Чем dataclass отличается от pydantic-модели?",
      """
`dataclass` — только генератор методов, типы не проверяются. Pydantic валидирует и приводит типы на входе.

```python
from dataclasses import dataclass
from pydantic import BaseModel, Field, ValidationError

@dataclass
class RawUser:
    age: int

print(RawUser("42").age)   # "42", silently wrong type

class User(BaseModel):
    age: int = Field(ge=0, le=150)

try:
    print(User(age="42").age)   # 42, coerced
    User(age=200)               # ValidationError
except ValidationError as exc:
    print(exc.errors())
```

Pydantic — на границе системы (JSON, API, env). Внутри — dataclass: дешевле и без магии.
"""),
)

# ---------------------------------------------------------------------------
# C++
# ---------------------------------------------------------------------------
add(
    E("cpp", "unique_ptr",
      "Покажи RAII на unique_ptr в C++.",
      """
```cpp
#include <memory>
#include <cstdio>

struct File {
    explicit File(const char* path) : fh(std::fopen(path, "rb")) {}
    ~File() { if (fh) std::fclose(fh); }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    std::FILE* fh;
};

void read_head(const char* path) {
    auto file = std::make_unique<File>(path);   // deleted automatically
    if (!file->fh) return;
    char buf[16];
    std::fread(buf, 1, sizeof buf, file->fh);
}
```

`unique_ptr` владеет объектом единолично и удаляет его в деструкторе, включая выход по исключению. Копирование запрещено, передача владения — `std::move`.
"""),
    E("cpp", "move semantics",
      "Что делает std::move и когда он не нужен?",
      """
`std::move` — это `static_cast<T&&>`, он не двигает байты, а разрешает выбрать move-конструктор.

```cpp
#include <string>
#include <utility>
#include <vector>

std::vector<std::string> make_names() {
    std::vector<std::string> out;
    std::string name = "extra-code";
    out.push_back(std::move(name));   // name is moved-from, empty but valid
    return out;                       // NRVO, no move needed
}

void consume(std::string s);

std::string value = "hello";
consume(std::move(value));            // ownership handed off
```

Не пиши `return std::move(local);` — это ломает NRVO. После move объект в валидном, но неопределённом состоянии: читать значение нельзя, присваивать можно.
"""),
    E("cpp", "templates",
      "Напиши шаблонную функцию с ограничением через concept.",
      """
```cpp
#include <concepts>
#include <vector>
#include <numeric>
#include <iostream>

template <std::floating_point T>
T mean(const std::vector<T>& values) {
    if (values.empty()) throw std::invalid_argument("empty");
    return std::accumulate(values.begin(), values.end(), T{0}) / values.size();
}

int main() {
    std::cout << mean(std::vector<double>{1.0, 2.0, 3.0}) << "\\n";
    // mean(std::vector<int>{1, 2});  // compile error: int is not floating_point
}
```

Concept проверяет требование на этапе компиляции и даёт внятную ошибку. До C++20 — `std::enable_if` и SFINAE, читается хуже.
"""),
    E("cpp", "stl algorithms",
      "Покажи типичную работу с STL algorithms вместо циклов.",
      """
```cpp
#include <algorithm>
#include <numeric>
#include <vector>
#include <iostream>

int main() {
    std::vector<int> v{5, 3, 8, 1, 9, 2};

    std::sort(v.begin(), v.end());
    auto it = std::lower_bound(v.begin(), v.end(), 5);   // O(log n)
    bool has_five = it != v.end() && *it == 5;

    std::vector<int> doubled;
    doubled.reserve(v.size());
    std::transform(v.begin(), v.end(), std::back_inserter(doubled),
                   [](int x) { return x * 2; });

    int total = std::accumulate(v.begin(), v.end(), 0);
    std::cout << has_five << ' ' << total << '\\n';
}
```

Алгоритмы выражают намерение и не дают промахнуться на единицу в индексах. `reserve` до `back_inserter` убирает лишние реаллокации.
"""),
    E("cpp", "lambda capture",
      "Как работают захваты в лямбдах C++ и где ловушка?",
      """
```cpp
#include <functional>
#include <iostream>
#include <memory>

int main() {
    int a = 1, b = 2;
    auto by_value = [a, b] { return a + b; };
    auto by_ref   = [&a, &b] { a += b; };
    auto init     = [n = 10] { return n * 2; };

    by_ref();
    std::cout << by_value() << ' ' << a << '\\n';   // 3 3
}
```

Ловушка — `[&]` в лямбде, которая живёт дольше своих переменных (сохранена в `std::function`, ушла в поток). Для таких случаев захватывай `shared_ptr` по значению: `[p = std::move(ptr)] { ... }`.
"""),
    E("cpp", "virtual destructor",
      "Почему базовому классу нужен виртуальный деструктор?",
      """
Удаление через указатель на базу без виртуального деструктора — undefined behavior: деструктор наследника не вызовется, ресурсы утекут.

```cpp
#include <memory>
#include <iostream>

struct Base {
    virtual ~Base() = default;
    virtual void run() const { std::cout << "base\\n"; }
};

struct Derived : Base {
    ~Derived() override { std::cout << "derived dtor\\n"; }
    void run() const override { std::cout << "derived\\n"; }
};

int main() {
    std::unique_ptr<Base> p = std::make_unique<Derived>();
    p->run();          // derived
}                       // derived dtor, then base dtor
```

Правило: если класс полиморфный (есть хотя бы одна виртуальная функция) — деструктор виртуальный. `override` ловит несовпадение сигнатуры.
"""),
    E("cpp", "const correctness",
      "Как правильно расставить const в C++?",
      """
`const` на параметре-ссылке обещает не менять объект; `const` после метода — не менять поля; `const` на возврате `const T*` запрещает писать через указатель.

```cpp
#include <string>
#include <vector>

class Buffer {
public:
    explicit Buffer(std::size_t n) : data_(n) {}

    std::size_t size() const noexcept { return data_.size(); }
    char& at(std::size_t i) { return data_.at(i); }
    const char& at(std::size_t i) const { return data_.at(i); }

    void append(const std::string& text) { data_.insert(data_.end(), text.begin(), text.end()); }

private:
    std::vector<char> data_;
};

void print(const Buffer& buf) {
    for (std::size_t i = 0; i < buf.size(); ++i) (void) buf.at(i);
}
```

Const-метод можно вызвать на const-объекте и на временном. Возвращать `const T` по значению бессмысленно — это мешает move.
"""),
    E("cpp", "threads",
      "Покажи потокобезопасную очередь на mutex и condition_variable.",
      """
```cpp
#include <condition_variable>
#include <mutex>
#include <optional>
#include <queue>

template <class T>
class BlockingQueue {
public:
    void push(T value) {
        {
            std::lock_guard<std::mutex> lock(m_);
            q_.push(std::move(value));
        }
        cv_.notify_one();
    }

    std::optional<T> pop() {
        std::unique_lock<std::mutex> lock(m_);
        cv_.wait(lock, [this] { return closed_ || !q_.empty(); });
        if (q_.empty()) return std::nullopt;   // closed and drained
        T value = std::move(q_.front());
        q_.pop();
        return value;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(m_);
            closed_ = true;
        }
        cv_.notify_all();
    }

private:
    std::queue<T> q_;
    std::mutex m_;
    std::condition_variable cv_;
    bool closed_ = false;
};
```

`wait` с предикатом защищает от ложных пробуждений. `notify` вызывай после отпускания мьютекса: разбудишь поток, который сразу возьмёт лок.
"""),
    E("cpp", "string_view",
      "Чем string_view опасен?",
      """
`std::string_view` не владеет памятью. Если исходная строка умерла или переехала, view станет висячим.

```cpp
#include <string>
#include <string_view>
#include <iostream>

std::string_view bad() {
    std::string local = "temporary";
    return local;                  // dangling
}

void ok(std::string_view sv) {
    std::cout << sv << '\\n';      // fine while caller's buffer is alive
}

int main() {
    std::string owned = "hello";
    ok(owned);
    ok(std::string_view(owned).substr(1));   // "ello"
}
```

Не храни `string_view` в полях долгоживущего объекта и не возвращай из функции, где строка локальная. Для владения — `std::string`.
"""),
    E("cpp", "structured bindings",
      "Что такое structured bindings и где они удобны?",
      """
```cpp
#include <map>
#include <string>
#include <tuple>
#include <iostream>

std::tuple<int, bool, std::string> parse() { return {42, true, "ok"}; }

int main() {
    auto [code, ok, message] = parse();
    std::cout << code << ' ' << ok << ' ' << message << '\\n';

    std::map<std::string, int> counts{{"a", 1}, {"b", 2}};
    for (const auto& [key, value] : counts) {
        std::cout << key << '=' << value << '\\n';
    }
}
```

Для tuple элементы идут по позиции, для struct — по порядку полей. `auto&` даёт ссылку, `auto` — копию; на больших значениях копия дорогая.
"""),
    E("cpp", "shared_ptr cycle",
      "Почему shared_ptr может утечь и как это исправить?",
      """
Два объекта, держащие `shared_ptr` друг на друга, не отпустят счётчик никогда. Один конец делают `weak_ptr`.

```cpp
#include <memory>
#include <iostream>

struct Node {
    std::string name;
    std::shared_ptr<Node> next;     // owning edge
    std::weak_ptr<Node> prev;       // non-owning back edge
    ~Node() { std::cout << "dtor " << name << '\\n'; }
};

int main() {
    auto a = std::make_shared<Node>(Node{"a"});
    auto b = std::make_shared<Node>(Node{"b"});
    a->next = b;
    b->prev = a;                    // no cycle
}
```

`weak_ptr::lock()` возвращает `shared_ptr` или пустой, если объект уже умер. Циклы ищутся глазами по графу владения, не сборщиком.
"""),
    E("cpp", "cmake",
      "Дай минимальный CMakeLists для C++20 проекта.",
      """
```cmake
cmake_minimum_required(VERSION 3.20)
project(extra_code LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(core src/core.cpp)
target_include_directories(core PUBLIC include)
target_compile_options(core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(app src/main.cpp)
target_link_libraries(app PRIVATE core)
```

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/app
```

MSVC вместо `-Wall -Wextra` понимает `/W4 /permissive-`; оборачивай через `if(MSVC)`. `CMAKE_BUILD_TYPE` работает в single-config генераторах, для Visual Studio — `--config Release`.
"""),
    E("cpp", "undefined behavior",
      "Назови частые источники UB в C++ и как их ловить.",
      """
Классика: выход за границы массива, разыменование null/висячего указателя, знаковое переполнение, чтение неинициализированной переменной, нарушение strict aliasing, гонка данных.

```cpp
int arr[4];
arr[4] = 1;              // UB: out of bounds
int x = INT_MAX + 1;     // UB: signed overflow
int y;                    // uninitialized
if (y == 0) {}            // UB: read of indeterminate value

int* p = new int(5);
delete p;
*p = 7;                   // UB: use after free
```

Ловля: `-fsanitize=address,undefined` для gcc/clang, `/fsanitize=address` для MSVC, `_GLIBCXX_ASSERTIONS`, `-D_ITERATOR_DEBUG_LEVEL=2`. Санитайзеры меняют раскладку памяти, но ошибки находят до продакшена.
"""),
    E("cpp", "rule of five",
      "Что такое rule of five и когда писать свои операции?",
      """
Если класс владеет ресурсом, нужны пять операций: деструктор, копирующий конструктор, копирующее присваивание, перемещающий конструктор, перемещающее присваивание. Либо явно удалить копирование.

```cpp
#include <cstddef>
#include <utility>

class Buffer {
public:
    explicit Buffer(std::size_t n) : data_(new char[n]), size_(n) {}
    ~Buffer() { delete[] data_; }

    Buffer(const Buffer& other) : data_(new char[other.size_]), size_(other.size_) {
        std::copy(other.data_, other.data_ + size_, data_);
    }
    Buffer& operator=(const Buffer& other) {
        if (this != &other) {
            Buffer tmp(other);
            swap(tmp);
        }
        return *this;
    }
    Buffer(Buffer&& other) noexcept : data_(other.data_), size_(other.size_) {
        other.data_ = nullptr;
        other.size_ = 0;
    }
    Buffer& operator=(Buffer&& other) noexcept {
        if (this != &other) { delete[] data_; data_ = other.data_; size_ = other.size_; other.data_ = nullptr; other.size_ = 0; }
        return *this;
    }
    void swap(Buffer& other) noexcept { std::swap(data_, other.data_); std::swap(size_, other.size_); }

private:
    char* data_;
    std::size_t size_;
};
```

Правило нуля лучше: пусть владеет `std::vector` или `unique_ptr`, и ничего из пяти писать не надо.
"""),
)

# ---------------------------------------------------------------------------
# Java
# ---------------------------------------------------------------------------
add(
    E("java", "interface vs abstract",
      "Чем интерфейс отличается от абстрактного класса в Java?",
      """
Класс наследует один абстрактный класс, но реализует много интерфейсов. Интерфейс описывает контракт, абстрактный класс может хранить состояние.

```java
interface Shape {
    double area();
    default String describe() { return "area=" + area(); }
}

abstract class BaseShape implements Shape {
    protected final String name;
    protected BaseShape(String name) { this.name = name; }
    @Override public String toString() { return name + ": " + describe(); }
}

final class Circle extends BaseShape {
    private final double r;
    Circle(double r) { super("circle"); this.r = r; }
    @Override public double area() { return Math.PI * r * r; }
}
```

С Java 8 у интерфейсов есть `default` и `static` методы, с 9 — `private`. Поля интерфейса всегда `public static final`.
"""),
    E("java", "generics",
      "Как работают generics в Java и что такое стирание типов?",
      """
Параметры типов стираются до `Object` (или до bound), поэтому `List<String>` и `List<Integer>` в рантайме один класс.

```java
import java.util.List;
import java.util.ArrayList;

class Box<T extends Comparable<T>> {
    private final T value;
    Box(T value) { this.value = value; }
    T get() { return value; }
    boolean greaterThan(Box<T> other) { return value.compareTo(other.get()) > 0; }
}

public class Main {
    public static void main(String[] args) {
        Box<Integer> a = new Box<>(2);
        Box<Integer> b = new Box<>(1);
        System.out.println(a.greaterThan(b));

        List<? extends Number> nums = new ArrayList<Integer>();
        // nums.add(1);          // compile error: producer, unknown subtype
        Number n = nums.get(0);  // ok
    }
}
```

Стирание запрещает `new T()`, `T[]`, `instanceof List<String>`. PECS: `? extends T` — читаешь, `? super T` — пишешь.
"""),
    E("java", "collections",
      "Когда брать HashMap, TreeMap и LinkedHashMap?",
      """
- `HashMap` — O(1) в среднем, порядок не гарантирован.
- `LinkedHashMap` — O(1), сохраняет порядок вставки (или доступа при `accessOrder=true`).
- `TreeMap` — O(log n), ключи отсортированы, есть `firstKey`, `headMap`, `floorEntry`.

```java
import java.util.*;

public class Maps {
    public static void main(String[] args) {
        Map<String, Integer> counts = new HashMap<>();
        counts.merge("a", 1, Integer::sum);
        counts.putIfAbsent("b", 0);
        counts.computeIfPresent("b", (k, v) -> v + 10);

        TreeMap<Integer, String> ordered = new TreeMap<>();
        ordered.put(3, "three");
        ordered.put(1, "one");
        System.out.println(ordered.firstEntry());   // 1=one
        System.out.println(ordered.floorEntry(2));  // 1=one
    }
}
```

Ключом должен быть immutable-объект с корректными `equals`/`hashCode`. Мутация ключа после вставки ломает `HashMap`.
"""),
    E("java", "streams",
      "Покажи типичный Stream API: filter, map, collect.",
      """
```java
import java.util.*;
import java.util.stream.Collectors;

record User(String name, int age, String city) {}

public class Demo {
    public static void main(String[] args) {
        List<User> users = List.of(
            new User("Ann", 23, "Moscow"),
            new User("Bob", 17, "Kazan"),
            new User("Cid", 31, "Moscow")
        );

        Map<String, List<String>> byCity = users.stream()
            .filter(u -> u.age() >= 18)
            .collect(Collectors.groupingBy(
                User::city,
                Collectors.mapping(User::name, Collectors.toList())
            ));

        double avg = users.stream().mapToInt(User::age).average().orElse(0.0);
        System.out.println(byCity);
        System.out.println(avg);
    }
}
```

Стрим ленивый: терминальная операция (`collect`, `count`, `anyMatch`) запускает конвейер. `parallelStream` не всегда быстрее — замеряй JMH.
"""),
    E("java", "try with resources",
      "Как правильно закрывать ресурсы в Java?",
      """
`try-with-resources` закрывает всё в обратном порядке и подавляет ошибки закрытия в `getSuppressed()`.

```java
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

public class Reader {
    static int countLines(String path) throws IOException {
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            int lines = 0;
            while (br.readLine() != null) lines++;
            return lines;
        }
    }

    public static void main(String[] args) {
        try {
            System.out.println(countLines("notes.txt"));
        } catch (IOException e) {
            System.err.println("read failed: " + e.getMessage());
        }
    }
}
```

Ресурс должен реализовывать `AutoCloseable`. Можно открывать несколько через `;` — закроются в обратном порядке.
"""),
    E("java", "executor",
      "Как запустить задачи в пуле потоков и дождаться результата?",
      """
```java
import java.util.concurrent.*;
import java.util.List;

public class PoolDemo {
    public static void main(String[] args) throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(4);
        try {
            List<Callable<Integer>> tasks = List.of(
                () -> 1 + 1,
                () -> 20 * 3,
                () -> 7
            );
            List<Future<Integer>> futures = pool.invokeAll(tasks);
            for (Future<Integer> f : futures) {
                System.out.println(f.get(2, TimeUnit.SECONDS));
            }
        } finally {
            pool.shutdown();
            if (!pool.awaitTermination(5, TimeUnit.SECONDS)) {
                pool.shutdownNow();
            }
        }
    }
}
```

`shutdown()` перестаёт принимать задачи, `awaitTermination` ждёт завершения, `shutdownNow` прерывает. Не создавай пул на задачу — это дорого.
"""),
    E("java", "records",
      "Что дают record и sealed в Java?",
      """
`record` — неизменяемый носитель данных: канонический конструктор, `equals`, `hashCode`, `toString`, аксессоры. `sealed` ограничивает наследников.

```java
sealed interface Event permits Login, Logout {}

record Login(String user, long at) implements Event {}
record Logout(String user, long at) implements Event {}

public class Demo {
    static String describe(Event e) {
        return switch (e) {
            case Login l -> l.user() + " logged in at " + l.at();
            case Logout l -> l.user() + " logged out";
        };
    }

    public static void main(String[] args) {
        System.out.println(describe(new Login("ann", 100)));
    }
}
```

`record` не заменяет обычный класс: поля финальные, сеттеров нет, наследование запрещено. Для валидации пиши компактный конструктор.
"""),
    E("java", "optional",
      "Как использовать Optional без антипаттернов?",
      """
`Optional` — для возвращаемого значения, когда результата может не быть. Не для полей и не для параметров.

```java
import java.util.Optional;

public class Repo {
    private final java.util.Map<String, String> data = java.util.Map.of("a", "Ann");

    Optional<String> find(String key) {
        return Optional.ofNullable(data.get(key));
    }

    public static void main(String[] args) {
        Repo repo = new Repo();
        String name = repo.find("a").orElse("unknown");
        repo.find("b").ifPresentOrElse(
            v -> System.out.println("found " + v),
            () -> System.out.println("missing")
        );
        // repo.find("a").get();  // avoid: throws when empty
    }
}
```

`orElse` вычисляется всегда, `orElseGet` — только при пустом. `get()` без проверки — тот же NPE, только с другим именем.
"""),
    E("java", "junit",
      "Напиши тест JUnit 5 с параметрами и исключением.",
      """
```java
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

class ParserTest {
    static int parse(String raw) {
        int v = Integer.parseInt(raw);
        if (v < 0) throw new IllegalArgumentException("negative");
        return v;
    }

    @ParameterizedTest
    @ValueSource(strings = {"0", "1", "42"})
    void parsesNonNegative(String raw) {
        assertEquals(Integer.parseInt(raw), parse(raw));
    }

    @Test
    void rejectsNegative() {
        var ex = assertThrows(IllegalArgumentException.class, () -> parse("-5"));
        assertEquals("negative", ex.getMessage());
    }
}
```

JUnit 5 — это `junit-jupiter`; в Maven подключи `junit-jupiter` с `test` scope. `assertThrows` возвращает исключение, удобно проверять сообщение.
"""),
    E("java", "build tools",
      "Чем Maven отличается от Gradle и как собрать jar?",
      """
Maven — декларативный XML и жёсткий жизненный цикл. Gradle — Groovy/Kotlin DSL, инкрементальная сборка и гибкость.

```xml
<!-- pom.xml -->
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>dev.extra</groupId>
  <artifactId>app</artifactId>
  <version>1.0.0</version>
  <properties>
    <maven.compiler.release>21</maven.compiler.release>
  </properties>
</project>
```

```bash
mvn -q package            # target/app-1.0.0.jar
mvn -q test
java -jar target/app-1.0.0.jar
```

```kotlin
// build.gradle.kts
plugins { application }
application { mainClass.set("dev.extra.Main") }
```

Для fat-jar в Maven нужен `maven-shade-plugin`, в Gradle — `shadow` plugin.
"""),
    E("java", "gc",
      "Как устроен сборщик мусора в JVM и какие флаги крутить?",
      """
JVM делит кучу на молодое поколение (Eden + survivor) и старое. Молодые объекты умирают быстро, поэтому minor GC частый и дешёвый; major GC дороже.

- G1 — дефолт с Java 9, балансирует паузы.
- ZGC / Shenandoah — паузы меньше миллисекунды на больших кучах.
- Serial — для маленьких куч и одного ядра.

```bash
java -Xms2g -Xmx2g -XX:+UseG1GC -XX:MaxGCPauseMillis=100 -Xlog:gc*:file=gc.log:time,uptime -jar app.jar
```

`-Xms` равен `-Xmx`, чтобы JVM не росла на старте. `-Xlog:gc*` покажет реальные паузы; без лога тюнинг — гадание. `System.gc()` в продакшене не вызывают.
"""),
    E("java", "equals hashCode",
      "Почему equals и hashCode надо переопределять вместе?",
      """
`HashMap` сначала сравнивает `hashCode`, потом `equals`. Равные объекты обязаны иметь равные хэши, иначе запись «теряется».

```java
import java.util.Objects;

final class User {
    private final String email;
    private final int age;

    User(String email, int age) { this.email = email; this.age = age; }

    @Override public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof User other)) return false;
        return age == other.age && email.equals(other.email);
    }

    @Override public int hashCode() {
        return Objects.hash(email, age);
    }
}
```

`instanceof User other` — pattern matching (Java 16+). Для record это уже сгенерировано. Мутабельные поля в `hashCode` — мина: ключ перестанет находиться.
"""),
)

# ---------------------------------------------------------------------------
# C
# ---------------------------------------------------------------------------
add(
    E("c", "pointers",
      "Объясни арифметику указателей в C на примере массива.",
      """
Указатель шагает на размер типа: `p + 1` сдвигает на `sizeof(*p)` байт, а не на один.

```c
#include <stdio.h>

int main(void) {
    int a[5] = {10, 20, 30, 40, 50};
    int *p = a;                 // decays to &a[0]

    printf("%d %d\\n", *p, *(p + 2));   // 10 30
    printf("%td\\n", (p + 4) - p);      // 4 elements

    for (int *it = a; it != a + 5; ++it) {
        printf("%d ", *it);
    }
    putchar('\\n');
}
```

`a + 5` — указатель за последним элементом, его можно сравнивать, но нельзя разыменовывать. Разница указателей даёт `ptrdiff_t`, а не `int`.
"""),
    E("c", "malloc realloc",
      "Как правильно выделять и расширять память через malloc/realloc?",
      """
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    size_t cap = 4, len = 0;
    int *data = malloc(cap * sizeof *data);
    if (!data) return 1;

    for (int i = 0; i < 10; ++i) {
        if (len == cap) {
            size_t next = cap * 2;
            int *tmp = realloc(data, next * sizeof *data);
            if (!tmp) { free(data); return 1; }   // old block still valid
            data = tmp;
            cap = next;
        }
        data[len++] = i * i;
    }

    for (size_t i = 0; i < len; ++i) printf("%d ", data[i]);
    putchar('\\n');
    free(data);
    return 0;
}
```

Никогда не присваивай `data = realloc(data, ...)`: при ошибке старый указатель потеряется и память утечёт. `sizeof *data` переживает смену типа.
"""),
    E("c", "structs",
      "Как объявить структуру и передать её в функцию?",
      """
```c
#include <stdio.h>
#include <string.h>

typedef struct {
    char name[32];
    int age;
} Person;

void birthday(Person *p) {          // pointer: no copy, can mutate
    p->age += 1;
}

Person make_person(const char *name, int age) {
    Person p = {0};
    snprintf(p.name, sizeof p.name, "%s", name);
    p.age = age;
    return p;                        // returned by value, fine for small structs
}

int main(void) {
    Person a = make_person("Ann", 30);
    birthday(&a);
    printf("%s %d\\n", a.name, a.age);
}
```

`typedef` убирает `struct` из каждого объявления. Большие структуры передавай по `const T *`; маленькие — по значению.
"""),
    E("c", "function pointers",
      "Покажи функцию qsort с компаратором.",
      """
```c
#include <stdio.h>
#include <stdlib.h>

static int cmp_int(const void *a, const void *b) {
    int x = *(const int *)a;
    int y = *(const int *)b;
    return (x > y) - (x < y);       // no overflow, unlike x - y
}

int main(void) {
    int v[] = {5, 1, 4, 2, 3};
    size_t n = sizeof v / sizeof v[0];

    qsort(v, n, sizeof v[0], cmp_int);
    for (size_t i = 0; i < n; ++i) printf("%d ", v[i]);
    putchar('\\n');
}
```

`(x > y) - (x < y)` даёт -1/0/1 без переполнения. Компаратор обязан быть строгим: `0` только для равных, иначе `qsort` уйдёт в UB.
"""),
    E("c", "strings",
      "Почему strcpy и strncpy опасны и чем заменить?",
      """
`strcpy` не знает про размер буфера, `strncpy` не гарантирует `\\0` и забивает остаток нулями.

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    char dst[8];
    const char *src = "abcdefghij";

    snprintf(dst, sizeof dst, "%s", src);   // always NUL-terminated
    printf("%s len=%zu\\n", dst, strlen(dst));

    char safe[8];
    size_t n = strlen(src);
    if (n >= sizeof safe) n = sizeof safe - 1;
    memcpy(safe, src, n);
    safe[n] = '\\0';
    puts(safe);
}
```

`snprintf` возвращает сколько хотелось записать, а не сколько влезло. Для копирования с известной длиной — `memcpy` + ручной `\\0`.
"""),
    E("c", "file io",
      "Как прочитать бинарный файл в C?",
      """
```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    FILE *fh = fopen("data.bin", "rb");
    if (!fh) { perror("fopen"); return 1; }

    if (fseek(fh, 0, SEEK_END) != 0) { fclose(fh); return 1; }
    long size = ftell(fh);
    if (size < 0) { fclose(fh); return 1; }
    rewind(fh);

    unsigned char *buf = malloc((size_t)size);
    if (!buf) { fclose(fh); return 1; }

    size_t got = fread(buf, 1, (size_t)size, fh);
    fclose(fh);
    if (got != (size_t)size) { free(buf); return 1; }   // short read

    free(buf);
    return 0;
}
```

`fread` возвращает число прочитанных элементов; короткое чтение — это ошибка или конец файла, проверяй отдельно через `feof`/`ferror`. На текстовых файлах Windows `"r"` включает CRLF-переводы, для байтов — `"rb"`.
"""),
    E("c", "bitwise",
      "Покажи работу с битовыми масками и флагами.",
      """
```c
#include <stdio.h>
#include <stdint.h>

enum {
    FLAG_READ  = 1u << 0,
    FLAG_WRITE = 1u << 1,
    FLAG_EXEC  = 1u << 2,
};

static int has(uint32_t flags, uint32_t bit) { return (flags & bit) != 0; }
static uint32_t set(uint32_t flags, uint32_t bit) { return flags | bit; }
static uint32_t clear(uint32_t flags, uint32_t bit) { return flags & ~bit; }

int main(void) {
    uint32_t f = 0;
    f = set(f, FLAG_READ);
    f = set(f, FLAG_WRITE);
    printf("%d %d %d\\n", has(f, FLAG_READ), has(f, FLAG_EXEC), (int)f);
    f = clear(f, FLAG_WRITE);
    printf("%u\\n", f);
}
```

Проверка — `(x & mask) != 0`, не `x & mask == mask`: приоритеты другие. Сдвиг на ширину типа и больше — UB, маскируй `& 31` для 32-битных.
"""),
    E("c", "headers",
      "Как правильно организовать заголовочный файл и .c?",
      """
```c
/* counter.h */
#ifndef COUNTER_H
#define COUNTER_H

#include <stddef.h>

typedef struct Counter Counter;

Counter *counter_new(void);
void counter_free(Counter *c);
void counter_add(Counter *c, size_t n);
size_t counter_value(const Counter *c);

#endif
```

```c
/* counter.c */
#include "counter.h"
#include <stdlib.h>

struct Counter { size_t value; };

Counter *counter_new(void) {
    Counter *c = malloc(sizeof *c);
    if (c) c->value = 0;
    return c;
}
void counter_free(Counter *c) { free(c); }
void counter_add(Counter *c, size_t n) { c->value += n; }
size_t counter_value(const Counter *c) { return c->value; }
```

Непрозрачный тип скрывает поля и позволяет менять структуру без пересборки всех клиентов. `#pragma once` короче, но не стандарт; include guards работают везде.
"""),
    E("c", "gcc sanitizers",
      "Какие флаги gcc включить для отладки и проверки памяти?",
      """
```bash
gcc -std=c17 -Wall -Wextra -Wpedantic -Wconversion \\
    -g -O1 -fsanitize=address,undefined -fno-omit-frame-pointer \\
    main.c -o app
./app
```

- `-Wall -Wextra` — базовые предупреждения.
- `-fsanitize=address` — выход за границы, use-after-free, утечки (`ASAN_OPTIONS=detect_leaks=1`).
- `-fsanitize=undefined` — знаковое переполнение, сдвиги, null.
- `-g` — символы для трейсбека.

Релиз собирай отдельно с `-O2 -DNDEBUG` и без санитайзеров: они замедляют в 2–10 раз.
"""),
    E("c", "linked list",
      "Реализуй односвязный список на C с добавлением и удалением.",
      """
```c
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int value;
    struct Node *next;
} Node;

static Node *push_front(Node *head, int value) {
    Node *n = malloc(sizeof *n);
    if (!n) return head;
    n->value = value;
    n->next = head;
    return n;
}

static Node *remove_value(Node *head, int value) {
    Node **link = &head;
    while (*link) {
        if ((*link)->value == value) {
            Node *dead = *link;
            *link = dead->next;
            free(dead);
            continue;
        }
        link = &(*link)->next;
    }
    return head;
}

static void free_list(Node *head) {
    while (head) { Node *next = head->next; free(head); head = next; }
}

int main(void) {
    Node *head = NULL;
    for (int i = 0; i < 5; ++i) head = push_front(head, i);
    head = remove_value(head, 2);
    for (Node *p = head; p; p = p->next) printf("%d ", p->value);
    free_list(head);
}
```

Указатель на указатель (`Node **link`) убирает отдельный случай удаления головы. Освобождай список до конца, иначе утечка.
"""),
    E("c", "undefined behavior",
      "Перечисли частые UB в C.",
      """
- Выход за границы массива и разыменование null.
- Знаковое переполнение (`INT_MAX + 1`).
- Чтение неинициализированной переменной.
- Нарушение strict aliasing (чтение `int` через `float*`).
- Двойной `free`, use-after-free.
- Сдвиг на отрицательное число или >= ширины типа.
- Изменение строкового литерала (`char *s = "abc"; s[0] = 'x';`).
- Отсутствие `return` в не-void функции, если результат используется.

```c
int f(void) { }            // UB when result is used
char *s = "literal";
s[0] = 'L';                // UB, may segfault

unsigned u = 0;
u--;                       // defined: wraps to UINT_MAX, not UB
```

Компилятор вправе предполагать, что UB не случается, и выкидывать проверки. Лечится `-fsanitize=undefined,address` и дисциплиной.
"""),
)

# ---------------------------------------------------------------------------
# C#
# ---------------------------------------------------------------------------
add(
    E("csharp", "properties",
      "Как объявить свойство с валидацией в C#?",
      """
```csharp
public sealed class User
{
    private string _name = "";
    private int _age;

    public required string Name
    {
        get => _name;
        set => _name = !string.IsNullOrWhiteSpace(value)
            ? value.Trim()
            : throw new ArgumentException("name is empty", nameof(value));
    }

    public int Age
    {
        get => _age;
        set => _age = value is >= 0 and <= 150
            ? value
            : throw new ArgumentOutOfRangeException(nameof(value));
    }
}

var u = new User { Name = "Ann", Age = 30 };
```

`required` (C# 11) заставляет задать свойство в инициализаторе. Автосвойства (`public int Age { get; set; }`) не дают валидации — для этого нужен backing field.
"""),
    E("csharp", "linq",
      "Покажи LINQ: фильтрация, группировка, проекция.",
      """
```csharp
using System;
using System.Linq;

record User(string Name, int Age, string City);

class Demo
{
    static void Main()
    {
        var users = new[]
        {
            new User("Ann", 23, "Moscow"),
            new User("Bob", 17, "Kazan"),
            new User("Cid", 31, "Moscow"),
        };

        var adults = users.Where(u => u.Age >= 18)
                          .OrderBy(u => u.Name)
                          .Select(u => u.Name)
                          .ToList();

        var byCity = users.GroupBy(u => u.City)
                          .Select(g => new { City = g.Key, Count = g.Count() });

        foreach (var row in byCity) Console.WriteLine($"{row.City}: {row.Count}");
    }
}
```

LINQ ленивый до терминальной операции (`ToList`, `Count`, `First`). Повторный проход по `IEnumerable` может заново ходить в базу — материализуй один раз.
"""),
    E("csharp", "async await",
      "Как правильно писать async/await в C#?",
      """
```csharp
using System;
using System.Net.Http;
using System.Threading.Tasks;

class Downloader
{
    private static readonly HttpClient Http = new();

    static async Task<string> FetchAsync(string url)
    {
        using var response = await Http.GetAsync(url);
        response.EnsureSuccessStatusCode();
        return await response.Content.ReadAsStringAsync();
    }

    static async Task Main()
    {
        var text = await FetchAsync("https://example.com");
        Console.WriteLine(text.Length);
    }
}
```

Не блокируй через `.Result`/`.Wait()` — это дедлок в UI/ASP.NET контексте. `async void` только для обработчиков событий; везде `async Task`. Суффикс `Async` — соглашение.
"""),
    E("csharp", "dispose",
      "Чем отличается using от финализатора и как реализовать IDisposable?",
      """
`using` вызывает `Dispose` детерминированно. Финализатор вызывает GC когда захочет, порядок не гарантирован.

```csharp
using System;
using System.IO;

public sealed class LogWriter : IDisposable
{
    private readonly StreamWriter _writer;
    private bool _disposed;

    public LogWriter(string path) => _writer = new StreamWriter(path, append: true);

    public void Write(string line)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _writer.WriteLine(line);
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _writer.Dispose();
    }
}

using (var log = new LogWriter("app.log"))
{
    log.Write("started");
}
```

`using var` (C# 8) закрывает в конце области видимости. Если есть неуправляемый ресурс — добавляй финализатор и `GC.SuppressFinalize(this)` в `Dispose`.
"""),
    E("csharp", "generics",
      "Как работают generics в C# и что такое ограничения?",
      """
В отличие от Java, generics в C# не стираются: для value type создаётся отдельный код, для reference type — общий.

```csharp
using System;
using System.Collections.Generic;

public static class Repo
{
    public static T Max<T>(IEnumerable<T> items) where T : IComparable<T>
    {
        using var e = items.GetEnumerator();
        if (!e.MoveNext()) throw new InvalidOperationException("empty");
        T best = e.Current;
        while (e.MoveNext())
            if (e.Current.CompareTo(best) > 0) best = e.Current;
        return best;
    }

    public static T Create<T>() where T : new() => new T();
}

Console.WriteLine(Repo.Max(new[] { 3, 9, 1 }));
```

Ограничения: `class`, `struct`, `new()`, `notnull`, базовый класс/интерфейс, `unmanaged`. Они дают компилятору знать, что можно вызывать.
"""),
    E("csharp", "records",
      "Что дают record и init в C#?",
      """
`record` — ссылочный тип с генерацией `Equals`, `GetHashCode`, `ToString`, деконструкции. `init` разрешает установку только в инициализаторе.

```csharp
using System;

public record Point(double X, double Y)
{
    public double Length => Math.Sqrt(X * X + Y * Y);
}

public sealed record User(string Name, int Age)
{
    public required string Email { get; init; }
}

class Demo
{
    static void Main()
    {
        var a = new Point(3, 4);
        var b = a with { X = 6 };          // non-destructive copy
        Console.WriteLine($"{a} {b} {b.Length}");

        var u = new User("Ann", 30) { Email = "a@b.c" };
        Console.WriteLine(u == u with { });   // value equality: True
    }
}
```

`record struct` даёт value type. `with` копирует все поля и меняет указанные — удобно для иммутабельных моделей.
"""),
    E("csharp", "events",
      "Как работают delegate и event в C#?",
      """
```csharp
using System;

public class Button
{
    public event EventHandler<string>? Clicked;

    public void Click() => Clicked?.Invoke(this, "clicked");
}

class Demo
{
    static void Main()
    {
        var button = new Button();
        button.Clicked += (_, message) => Console.WriteLine($"handler 1: {message}");
        button.Clicked += (_, message) => Console.WriteLine($"handler 2: {message}");
        button.Click();
    }
}
```

`event` снаружи класса можно только подписывать/отписывать, вызывать — только внутри. `?.Invoke` защищает от пустого списка подписчиков. Отписывайся (`-=`), если подписчик живёт дольше издателя, иначе утечка.
"""),
    E("csharp", "collections",
      "Когда брать Dictionary, HashSet и List в C#?",
      """
```csharp
using System;
using System.Collections.Generic;
using System.Linq;

class Demo
{
    static void Main()
    {
        var counts = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        foreach (var word in new[] { "a", "A", "b" })
            counts[word] = counts.GetValueOrDefault(word) + 1;

        var seen = new HashSet<int>();
        Console.WriteLine(seen.Add(1));   // True
        Console.WriteLine(seen.Add(1));   // False

        var list = counts.OrderByDescending(kv => kv.Value).ToList();
        Console.WriteLine(string.Join(", ", list.Select(kv => $"{kv.Key}={kv.Value}")));
    }
}
```

`Dictionary` — O(1) поиск по ключу, `HashSet` — уникальность, `List` — индексный доступ. `GetValueOrDefault` не бросает на отсутствии, в отличие от индексатора.
"""),
    E("csharp", "dotnet cli",
      "Как создать, собрать и запустить проект через dotnet CLI?",
      """
```bash
dotnet new console -n App -o src/App
cd src/App
dotnet add package System.Text.Json
dotnet build -c Release
dotnet run -c Release
dotnet publish -c Release -r win-x64 --self-contained false -o out
```

```xml
<!-- App.csproj -->
<Project Sdk="Microsoft.NET.Sdk">
  <PropertyGroup>
    <OutputType>Exe</OutputType>
    <TargetFramework>net8.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
  </PropertyGroup>
</Project>
```

`dotnet publish` кладёт зависимости рядом; `--self-contained true` тащит рантайм, файл больше, но .NET ставить не надо. `Nullable enable` включает проверку null на этапе компиляции.
"""),
    E("csharp", "nullable",
      "Что даёт nullable reference types?",
      """
Компилятор отслеживает, может ли ссылка быть null, и предупреждает о разыменовании.

```csharp
#nullable enable
using System;

class Demo
{
    static string? FindName(int id) => id == 1 ? "Ann" : null;

    static void Main()
    {
        string? name = FindName(2);
        Console.WriteLine(name?.Length ?? 0);      // safe

        string sure = name ?? "unknown";            // non-null after ??
        Console.WriteLine(sure.Length);

        // string bad = FindName(2);                // warning CS8600
        // Console.WriteLine(bad.Length);           // warning CS8602
    }
}
```

`?` — nullable, `!` — подавление предупреждения (обещание, что не null). Предупреждения, а не ошибки: включай `TreatWarningsAsErrors` в CI.
"""),
    E("csharp", "span",
      "Зачем нужен Span<T> в C#?",
      """
`Span<T>` — срез массива или памяти без копирования и без аллокации в куче.

```csharp
using System;

class Demo
{
    static int Sum(ReadOnlySpan<int> values)
    {
        int total = 0;
        foreach (var v in values) total += v;
        return total;
    }

    static void Main()
    {
        int[] data = { 1, 2, 3, 4, 5 };
        Console.WriteLine(Sum(data));            // implicit conversion
        Console.WriteLine(Sum(data.AsSpan(1, 3))); // 2+3+4

        Span<byte> buf = stackalloc byte[16];
        buf[0] = 42;
        Console.WriteLine(buf[0]);
    }
}
```

`Span<T>` живёт только на стеке: нельзя в поле класса или в `async`-методе. Для таких случаев — `Memory<T>`.
"""),
)

# ---------------------------------------------------------------------------
# x86-64 assembly
# ---------------------------------------------------------------------------
add(
    E("asm", "registers",
      "Какие регистры есть в x86-64 и за что они отвечают?",
      """
- `rax` — аккумулятор, возвращаемое значение, номер syscall.
- `rbx`, `r12`–`r15` — callee-saved, функция обязана сохранить.
- `rcx`, `rdx`, `rsi`, `rdi`, `r8`–`r11` — caller-saved, аргументы и временные.
- `rsp` — вершина стека, `rbp` — база кадра.
- `rip` — указатель инструкции.

Младшие части: `eax` (32), `ax` (16), `al`/`ah` (8). Запись в `eax` обнуляет старшие 32 бита `rax` — это важно для оптимизаций.

```asm
mov rax, 60        ; sys_exit
xor rdi, rdi       ; status 0
syscall
```

`rsp` меняется только парами push/pop или `sub`/`add`, иначе стек ломается.
"""),
    E("asm", "sysv calling convention",
      "Как передаются аргументы функции в System V AMD64?",
      """
Первые шесть целочисленных аргументов: `rdi, rsi, rdx, rcx, r8, r9`. Остальные — через стек справа налево. Возврат — в `rax` (и `rdx` для 128-бит).

```asm
; long add3(long a, long b, long c)  ->  rdi, rsi, rdx
add3:
    lea rax, [rdi + rsi]
    add rax, rdx
    ret

; caller
mov rdi, 1
mov rsi, 2
mov rdx, 3
call add3
; rax = 6
```

Плавающие аргументы идут в `xmm0`–`xmm7`. Перед `call` стек должен быть выровнен на 16 байт: на входе в функцию `rsp % 16 == 8` из-за адреса возврата.
"""),
    E("asm", "prologue",
      "Зачем нужен пролог функции и как выглядит эпилог?",
      """
Пролог сохраняет `rbp` вызывающего, строит кадр и выделяет локальные переменные. Эпилог возвращает всё назад.

```asm
func:
    push rbp            ; save caller's frame pointer
    mov rbp, rsp        ; new frame
    sub rsp, 32         ; locals, keep 16-byte alignment
    mov qword [rbp-8], 42

    mov rax, [rbp-8]
    leave               ; mov rsp, rbp; pop rbp
    ret
```

`leave` — короткая форма эпилога. Если локальных переменных нет и `rbp` не нужен, компилятор с `-O2` часто обходится без кадра и адресуется через `rsp` или `rip`.
"""),
    E("asm", "linux syscalls",
      "Как сделать write и exit через syscall в Linux x86-64?",
      """
Номер syscall — в `rax`, аргументы — `rdi, rsi, rdx, r10, r8, r9`, инструкция `syscall`, результат в `rax`.

```asm
; write(1, msg, len)
mov rax, 1          ; SYS_write
mov rdi, 1          ; stdout
lea rsi, [rel msg]
mov rdx, msg_len
syscall

; exit(0)
mov rax, 60         ; SYS_exit
xor rdi, rdi
syscall

section .rodata
msg: db "hello", 10
msg_len equ $ - msg
```

`r10` заменяет `rcx` в syscall-вызовах, потому что `syscall` портит `rcx` и `r11`. Отрицательное значение в `rax` — `-errno`.
"""),
    E("asm", "loop array",
      "Напиши цикл суммирования массива на ассемблере.",
      """
```asm
; long sum(const long *a, size_t n)   rdi = a, rsi = n
sum:
    xor eax, eax            ; total = 0
    xor ecx, ecx            ; i = 0
.loop:
    cmp rcx, rsi
    jae .done              ; unsigned i >= n
    add rax, [rdi + rcx*8]
    inc rcx
    jmp .loop
.done:
    ret
```

`jae` — unsigned, для счётчика размера это правильно. Индексная адресация `[base + index*scale]` допускает scale 1/2/4/8; шаг в байтах считается автоматически.
"""),
    E("asm", "movzx",
      "Чем отличаются mov, movzx и movsx?",
      """
- `mov` копирует биты как есть; при переходе в меньшую часть старшие биты теряются.
- `movzx` расширяет нулями.
- `movsx` расширяет знаковым битом.

```asm
mov  al, 0xFF        ; al = 0xFF, старшие биты rax не тронуты
movzx eax, al        ; eax = 0x000000FF
movsx eax, al        ; eax = 0xFFFFFFFF (-1)
movsxd rax, eax      ; sign-extend 32 -> 64
```

Для `char` без знака бери `movzx`, для `signed char` — `movsx`. Ошибка здесь даёт «магические» отрицательные числа.
"""),
    E("asm", "flags jcc",
      "Как работают флаги и условные переходы?",
      """
Сравнение — это вычитание без записи результата; флаги отражают результат.

- `ZF` — результат ноль.
- `SF` — знак.
- `CF` — беззнаковый перенос/заём.
- `OF` — знаковое переполнение.

```asm
cmp rax, rbx
je  equal        ; ZF=1
jne not_equal
jl  less_signed  ; SF != OF
jb  less_unsigned; CF=1
jg  greater_signed
ja  greater_unsigned
```

`jl/jg` — для знаковых, `jb/ja` — для беззнаковых. Перепутать — классическая ошибка при разборе чужого кода.
"""),
    E("asm", "stack alignment",
      "Почему стек должен быть выровнен на 16 байт перед call?",
      """
ABI требует 16-байтового выравнивания `rsp` в момент передачи управления. `call` кладёт 8-байтовый адрес возврата, поэтому на входе в функцию `rsp % 16 == 8`.

```asm
; before call: rsp % 16 == 0
push rbp            ; rsp % 16 == 8
mov rbp, rsp
sub rsp, 24         ; 8 + 24 = 32, still 16-aligned for calls
call other          ; callee sees rsp % 16 == 8

mov rsp, rbp
pop rbp
ret
```

Невыровненный стек ломает `movaps`/SSE и `printf` с `%f` — падение или мусор. Отладчик покажет `SIGSEGV` на инструкции, которой нет в исходнике.
"""),
    E("asm", "rip relative",
      "Что такое RIP-relative адресация и зачем она для PIC?",
      """
Адрес считается от текущей инструкции: `[rip + offset]`. Код остаётся рабочим при загрузке по любому базовому адресу.

```asm
lea rax, [rip + message]    ; address of message
mov rdi, rax
call puts

section .rodata
message: db "hello", 0
```

Абсолютные адреса в 64-битном коде требуют релокаций и ломают PIE. `lea` вместо `mov` не читает память — просто считает адрес.
"""),
    E("asm", "objdump",
      "Как читать дизассемблер через objdump?",
      """
```bash
objdump -d -M intel app | less
objdump -d -M intel --disassemble=main app
objdump -t app | grep ' F '
readelf -h app
```

Типичная строка:

```text
0000000000001149 <main>:
    1149:  push   rbp
    114a:  mov    rbp,rsp
    114d:  sub    rsp,0x10
    1151:  mov    DWORD PTR [rbp-0x4],0x2a
    1158:  mov    eax,DWORD PTR [rbp-0x4]
    115b:  leave
    115c:  ret
```

Слева адрес и байты, справа инструкция. `-M intel` переключает синтаксис с AT&T на привычный Intel. `--disassemble=` фильтрует по символу.
"""),
    E("asm", "windows x64 abi",
      "Чем Windows x64 ABI отличается от System V?",
      """
- Аргументы: `rcx, rdx, r8, r9`, дальше стек.
- 32 байта shadow space на стеке обязан выделить вызывающий.
- `xmm0`–`xmm3` для float, `rax` — возврат.
- `rbx, rbp, rdi, rsi, r12`–`r15` — callee-saved.
- Нет red zone.

```asm
; void f(int a, int b)
f:
    sub rsp, 40          ; 32 shadow + 8 align
    mov [rsp+32], ecx    ; spill a if needed
    mov [rsp+40], edx
    add rsp, 40
    ret
```

Забыл shadow space — callee запишет туда свои временные и испортит твои данные. Отладка такого «иногда падает» дорогая.
"""),
    E("asm", "intrinsics",
      "Как использовать SIMD без ассемблера?",
      """
Через intrinsics: `immintrin.h`, типы `__m256`, функции `_mm256_*`. Компилятор сам выбирает регистры.

```c
#include <immintrin.h>
#include <stdio.h>

int main(void) {
    float a[8] = {1,2,3,4,5,6,7,8};
    float b[8] = {8,7,6,5,4,3,2,1};
    float out[8];

    __m256 va = _mm256_loadu_ps(a);
    __m256 vb = _mm256_loadu_ps(b);
    _mm256_storeu_ps(out, _mm256_add_ps(va, vb));

    for (int i = 0; i < 8; ++i) printf("%.0f ", out[i]);
}
```

Собирай с `-mavx2` (или `-march=native`), иначе intrinsic не скомпилируется. `loadu` — без требования выравнивания, `load` требует 32 байта.
"""),
)

# ---------------------------------------------------------------------------
# Reverse engineering
# ---------------------------------------------------------------------------
add(
    E("re", "pe structure",
      "Из чего состоит PE-файл Windows?",
      """
- `DOS header` (`MZ`) → `e_lfanew` указывает на NT headers.
- `NT headers`: `Signature` (`PE\\0\\0`), `FileHeader` (machine, number of sections), `OptionalHeader` (entry point `AddressOfEntryPoint`, `ImageBase`, data directories).
- `Section table`: `.text` (код), `.rdata` (константы), `.data` (данные), `.rsrc` (ресурсы), `.reloc` (релокации).
- Data directories: импорт, экспорт, TLS, исключения, сертификат.

```python
import pefile

pe = pefile.PE("app.exe")
print(hex(pe.OPTIONAL_HEADER.ImageBase))
print(hex(pe.OPTIONAL_HEADER.AddressOfEntryPoint))
for s in pe.sections:
    print(s.Name.decode().rstrip("\\x00"), hex(s.VirtualAddress), s.Characteristics)
```

RVA — смещение в памяти, file offset — в файле. Перевод: `offset = RVA - VirtualAddress + PointerToRawData` для секции, куда попадает RVA.
"""),
    E("re", "imports iat",
      "Что такое IAT и как по импортам понять поведение программы?",
      """
IAT (Import Address Table) — массив указателей на функции DLL, заполняется загрузчиком. Список импортов — самый быстрый способ понять, что программа делает.

```python
import pefile

pe = pefile.PE("app.exe")
if hasattr(pe, "DIRECTORY_ENTRY_IMPORT"):
    for entry in pe.DIRECTORY_ENTRY_IMPORT:
        dll = entry.dll.decode()
        names = [imp.name.decode() if imp.name else f"ord#{imp.ordinal}" for imp in entry.imports]
        print(dll, names[:12])
```

`VirtualProtect` + `WriteProcessMemory` + `CreateRemoteThread` — инъекция. `CryptEncrypt`, `BCryptDecrypt` — крипта. `IsDebuggerPresent` — анти-отладка. `RegSetValueEx` — автозапуск.
"""),
    E("re", "workflow",
      "Опиши рабочий процесс статического анализа в Ghidra/IDA.",
      """
1. Определить тип файла: `file`, `Detect It Easy`, `pefile`.
2. Открыть в Ghidra (бесплатно) или IDA, дать автоанализ.
3. Найти `entry`, `main`, `WinMain`; для C++ — `main` через `__scrt_common_main_seh`.
4. Пройтись по строкам (`Strings`, Shift+F12) и перекрёстным ссылкам (XREF).
5. Переименовать функции по смыслу, задать типы структур.
6. Читать импорты и вызовы API — восстановить сценарий.
7. Динамика: брейкпоинты в x64dbg на подозрительных вызовах, смотреть аргументы.

```text
Ghidra: File > Import File, Analyze, Window > Defined Strings
IDA:    Shift+F12 strings, X xrefs, N rename, Y retype
```

Статика отвечает «что», динамика — «с какими данными». Начинай со строк и импортов: 80% поведения видно до первой инструкции.
"""),
    E("re", "x64dbg",
      "Как работать в x64dbg: брейкпоинты и стек?",
      """
- `F2` — breakpoint на инструкции, `F9` — запуск, `F7` — шаг в, `F8` — шаг через.
- `Ctrl+G` — переход по адресу/символу.
- Вкладка `Call Stack` — цепочка возвратов; двойной клик переходит в кадр.
- `Follow in Dump` на регистре/стеке показывает память.
- `ScyllaHide` прячет отладчик от `IsDebuggerPresent`/PEB-проверок.

```text
bp kernel32.CreateFileW
bp 0x401000
run
```

На входе в функцию Windows x64 аргументы уже в `rcx, rdx, r8, r9`; строка — указатель в `rcx`, смотри её в дампе. `bp` на API ловит все вызовы, условие `bp addr, "strcmp(rcx, \"admin\")==0"` сужает поток.
"""),
    E("re", "finding main",
      "Как найти main в stripped-бинарнике?",
      """
1. Точка входа (`entry`) обычно `_start`, который вызывает `__libc_start_main`. Первый аргумент — адрес `main`.
2. В Windows ищи вызов `__scrt_common_main_seh` из `mainCRTStartup`.
3. По строкам: найди сообщение и XREF на него.
4. По импортам: `GetCommandLineA`/`argv` рядом с `main`.

```asm
; linux _start
xor ebp, ebp
mov r9, rdx
pop rsi
mov rdx, rsp
lea rdi, [rip + main]     ; <- this is main
call __libc_start_main
```

В x64dbg/GDB ставь брейкпоинт на `__libc_start_main` и читай `rdi`. В IDA — `entry` и первый `call` с адресом в `rdi`.
"""),
    E("re", "strings xrefs",
      "Как использовать строки и перекрёстные ссылки при реверсе?",
      """
Строка — якорь: сообщение об ошибке, имя файла, URL, формат. Найдя её адрес, смотришь, кто его читает.

```text
Shift+F12  — список строк
Ctrl+F     — поиск по тексту
X          — xrefs на строку
```

```asm
lea rdi, [rip + aInvalidKey]   ; "invalid key"
call puts
```

Если строка используется в проверке, рядом будет ветвление `test`/`cmp` + `jz`. Меняй `jz` на `jnz` в отладчике — проверишь гипотезу, не патча файл. Для Unicode в IDA — `Strings > Unicode`.
"""),
    E("re", "upx",
      "Как распаковать UPX и что делать с кастомным пакером?",
      """
UPX — самый простой случай: секции `UPX0`, `UPX1`, высокая энтропия, `pushad` в начале.

```bash
upx -d packed.exe -o unpacked.exe
```

Если UPX модифицирован (битый заголовок), правят поле `p_filesize`/`p_blocksize` или снимают дамп после распаковки: брейкпоинт на `popad`, затем `OEP` (original entry point), `Scylla` для восстановления IAT.

Кастомный пакер: запусти, дождись распаковки, поставь hardware breakpoint на `Execute` для секции `.text`, сними дамп через `Scylla`/`PE dump`, почини импорты. Энтропия > 7.5 и мало импортов — признак упаковки.
"""),
    E("re", "anti debug",
      "Как программа обнаруживает отладчик и как это обойти?",
      """
Классика:

- `IsDebuggerPresent()` — читает `PEB->BeingDebugged`.
- `CheckRemoteDebuggerPresent()`.
- `NtQueryInformationProcess(ProcessDebugPort)`.
- `rdtsc` — замер времени между инструкциями.
- `int 3`/`int 2d` и обработка исключений.
- Проверка `PEB->NtGlobalFlag` (флаги кучи отладчика).

```c
if (IsDebuggerPresent()) ExitProcess(0);
```

Обход: ScyllaHide, патч `BeingDebugged` в PEB, брейкпоинт на `ExitProcess` и подмена возврата, `x64dbg` plugin `HideDebugger`. Для учебного разбора достаточно `ScyllaHide` с профилем VMProtect.
"""),
    E("re", "api hooking",
      "Как работает API-хукинг через IAT и inline?",
      """
IAT-хук: заменить указатель в таблице импорта на свою функцию. Inline-хук: записать `jmp` в начало функции.

```cpp
// IAT hook sketch: swap MessageBoxW pointer in the import table
#include <windows.h>

using MsgBoxFn = int (WINAPI*)(HWND, LPCWSTR, LPCWSTR, UINT);
MsgBoxFn original = nullptr;

int WINAPI Hooked(HWND h, LPCWSTR text, LPCWSTR caption, UINT type) {
    return original(h, L"hooked", caption, type);
}

void patch_iat(HMODULE module, const char* dll, const char* name, void* replacement) {
    auto base = reinterpret_cast<unsigned char*>(module);
    auto dos = reinterpret_cast<IMAGE_DOS_HEADER*>(base);
    auto nt = reinterpret_cast<IMAGE_NT_HEADERS*>(base + dos->e_lfanew);
    auto& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    auto imp = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + dir.VirtualAddress);
    for (; imp->Name; ++imp) {
        if (_stricmp(reinterpret_cast<char*>(base + imp->Name), dll) != 0) continue;
        auto thunk = reinterpret_cast<IMAGE_THUNK_DATA*>(base + imp->FirstThunk);
        for (; thunk->u1.Function; ++thunk) {
            auto& slot = reinterpret_cast<void*&>(thunk->u1.Function);
            if (_stricmp(reinterpret_cast<char*>(base + thunk->u1.AddressOfData + 2), name) == 0 &&
                slot != replacement) {
                DWORD old = 0;
                VirtualProtect(&slot, sizeof slot, PAGE_READWRITE, &old);
                original = reinterpret_cast<MsgBoxFn>(slot);
                slot = replacement;
                VirtualProtect(&slot, sizeof slot, old, &old);
                return;
            }
        }
    }
}
```

Inline-хук: `VirtualProtect` на страницу, сохранить первые 5+ байт, записать `E9 rel32` (x86) или `FF 25` trampoline (x64), в trampoline — сохранённые байты и прыжок назад. На x64 безопаснее Detours/MinHook: они учитывают длину инструкций.
"""),
    E("re", "pattern scan",
      "Как найти адрес функции по сигнатуре в памяти процесса?",
      """
Сигнатура — последовательность байт с масками, устойчивая к обновлениям. Ищут по образу модуля, адрес = база + смещение.

```cpp
#include <windows.h>
#include <vector>
#include <cstdint>

uintptr_t find_pattern(uintptr_t base, size_t size, const std::vector<int>& pattern) {
    auto data = reinterpret_cast<const uint8_t*>(base);
    for (size_t i = 0; i + pattern.size() <= size; ++i) {
        bool ok = true;
        for (size_t j = 0; j < pattern.size(); ++j) {
            if (pattern[j] != -1 && data[i + j] != static_cast<uint8_t>(pattern[j])) { ok = false; break; }
        }
        if (ok) return base + i;
    }
    return 0;
}

// "48 8B 05 ? ? ? ? 48 85 C0" -> {0x48,0x8B,0x05,-1,-1,-1,-1,0x48,0x85,0xC0}
```

Бери сигнатуру из уникальной окрестности функции, без относительных адресов (`?`), иначе патч сломает поиск. Сканируй только исполняемые секции модуля.
"""),
    E("re", "vtable",
      "Как восстановить классы C++ по таблицам виртуальных функций?",
      """
У полиморфного объекта первые 8 байт — указатель на vtable. В vtable лежат указатели на виртуальные методы в порядке объявления.

```text
obj -> [vptr] -> [ &dtor, &method1, &method2, ... ]
```

```python
# sketch: read vptr and first slots from a process dump
import struct
vptr = struct.unpack_from("<Q", data, obj_off)[0]
slots = struct.unpack_from("<QQQ", data, vptr - image_base + file_off)
```

В IDA/Ghidra: `Create structure`, поле `void **vftable`, затем `Force call type` на слотах. RTTI (`type_info`) при включённых исключениях даёт имя класса — ищи по строке `.?AV...@@` (MSVC) или `_ZTV` (Itanium).
"""),
    E("re", "dotnet",
      "Как реверсить .NET-сборку?",
      """
.NET компилируется в IL, а не в машинный код: восстанавливается почти исходник.

```bash
# ILSpy CLI / dnSpyEx (GUI)
ilspycmd app.dll -o out
# or: dnSpy -> open exe -> tree of namespaces -> decompile
```

Смотри `Assembly-CSharp.dll` в Unity-играх: там логика. Обфускация (ConfuserEx) ломает имена и control flow — помогает `de4dot`, затем ручная правка в dnSpy. Правка IL сохраняется обратно в сборку (`File > Save Module`).
"""),
    E("re", "static vs dynamic",
      "Когда хватает статического анализа, а когда нужен отладчик?",
      """
Статика: понять структуру, алгоритм, константы, строки, импорты. Быстро, безопасно, но упирается в упаковку и генерацию кода в рантайме.

Динамика: узнать реальные аргументы, распакованный код, сетевые адреса, ключи. Медленнее, требует изоляции.

Практика: начни со статики (strings, imports, Ghidra auto-analysis), гипотезы проверяй в x64dbg. Для вредоносного кода — только VM без сети, snapshots, `Procmon`/`Wireshark`. Никогда не запускай образец на рабочей машине.
"""),
    E("re", "stack frame",
      "Как по стеку понять цепочку вызовов в дизассемблере?",
      """
В кадре `[rbp]` — сохранённый `rbp` вызывающего, `[rbp+8]` — адрес возврата, `[rbp+16]` — первый стековый аргумент.

```text
rbp+16  arg7
rbp+8   return address  -> caller's rbp chain
rbp     saved rbp
rbp-8   local
```

```gdb
bt            # backtrace
info frame
x/8gx $rsp
```

В оптимизированном коде `rbp` может не использоваться (frame pointer omission) — тогда ориентируйся на `rsp` и `.pdata`/`.eh_frame`. В x64dbg вкладка Call Stack делает это автоматически.
"""),
)

# ---------------------------------------------------------------------------
# Algorithms and data structures
# ---------------------------------------------------------------------------
add(
    E("algo", "binary search",
      "Напиши бинарный поиск без ошибок на единицу.",
      """
```python
from typing import Sequence

def lower_bound(a: Sequence[int], target: int) -> int:
    lo, hi = 0, len(a)          # [lo, hi)
    while lo < hi:
        mid = (lo + hi) // 2
        if a[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo                    # first index with a[i] >= target

print(lower_bound([1, 3, 3, 7], 3))   # 1
print(lower_bound([1, 3, 3, 7], 4))   # 3
```

Полуинтервал `[lo, hi)` убирает особые случаи. `(lo + hi) // 2` безопасен в Python; в C++ для больших индексов — `lo + (hi - lo) / 2`.
"""),
    E("algo", "quicksort",
      "Покажи быструю сортировку и её сложность.",
      """
```python
def quicksort(a: list[int]) -> list[int]:
    if len(a) <= 1:
        return a
    pivot = a[len(a) // 2]
    less = [x for x in a if x < pivot]
    equal = [x for x in a if x == pivot]
    greater = [x for x in a if x > pivot]
    return quicksort(less) + equal + quicksort(greater)
```

Среднее O(n log n), худшее O(n²) при неудачном пивоте, память O(n) на списки. In-place версия экономит память, но требует аккуратного разбиения Ломуто/Хоара. Стабильности нет; для гарантии O(n log n) — `sorted` (Timsort) или heapsort.
"""),
    E("algo", "hash table",
      "Как устроена хэш-таблица и что такое коллизии?",
      """
Массив бакетов; индекс = `hash(key) % capacity`. Коллизии решают цепочками (chaining) или открытой адресацией.

```python
class HashMap:
    def __init__(self, capacity: int = 8) -> None:
        self._buckets: list[list[tuple[str, int]]] = [[] for _ in range(capacity)]
        self._size = 0

    def _index(self, key: str) -> int:
        return hash(key) % len(self._buckets)

    def put(self, key: str, value: int) -> None:
        bucket = self._buckets[self._index(key)]
        for i, (k, _) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return
        bucket.append((key, value))
        self._size += 1

    def get(self, key: str) -> int | None:
        for k, v in self._buckets[self._index(key)]:
            if k == key:
                return v
        return None
```

При load factor > 0.7 таблицу расширяют и перехэшируют: O(n) разово, амортизированно O(1). Плохая хэш-функция превращает всё в O(n).
"""),
    E("algo", "bfs",
      "Напиши BFS и объясни, зачем очередь.",
      """
```python
from collections import deque

def bfs(graph: dict[str, list[str]], start: str) -> dict[str, int]:
    dist = {start: 0}
    q = deque([start])
    while q:
        node = q.popleft()
        for nxt in graph.get(node, []):
            if nxt not in dist:
                dist[nxt] = dist[node] + 1
                q.append(nxt)
    return dist

graph = {"a": ["b", "c"], "b": ["d"], "c": ["d"], "d": []}
print(bfs(graph, "a"))   # {'a': 0, 'b': 1, 'c': 1, 'd': 2}
```

Очередь даёт порядок по слоям, поэтому первое посещение — кратчайший путь в невзвешенном графе. `deque.popleft()` — O(1); `list.pop(0)` — O(n).
"""),
    E("algo", "dfs",
      "Покажи DFS рекурсивно и итеративно.",
      """
```python
def dfs_recursive(graph: dict[str, list[str]], node: str, seen: set[str] | None = None) -> list[str]:
    seen = seen or set()
    seen.add(node)
    order = [node]
    for nxt in graph.get(node, []):
        if nxt not in seen:
            order += dfs_recursive(graph, nxt, seen)
    return order

def dfs_iterative(graph: dict[str, list[str]], start: str) -> list[str]:
    seen, order, stack = set(), [], [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        order.append(node)
        stack.extend(reversed(graph.get(node, [])))
    return order
```

Рекурсия короче, но глубина графа ограничена стеком. Итеративная версия с явным стеком не падает на 10⁵ вершинах. `reversed` сохраняет порядок обхода как в рекурсии.
"""),
    E("algo", "dp knapsack",
      "Реши задачу о рюкзаке 0/1 динамикой.",
      """
```python
def knapsack(weights: list[int], values: list[int], capacity: int) -> int:
    dp = [0] * (capacity + 1)
    for w, v in zip(weights, values):
        for cap in range(capacity, w - 1, -1):   # backwards: one item once
            dp[cap] = max(dp[cap], dp[cap - w] + v)
    return dp[capacity]

print(knapsack([2, 3, 4], [3, 4, 5], 5))   # 7
```

O(n·W) времени и O(W) памяти. Обратный проход по ёмкости не даёт взять предмет дважды; для неограниченного рюкзака — прямой проход.
"""),
    E("algo", "reverse list",
      "Разверни односвязный список итеративно.",
      """
```python
class Node:
    __slots__ = ("value", "next")
    def __init__(self, value: int, nxt: "Node | None" = None) -> None:
        self.value = value
        self.next = nxt

def reverse(head: Node | None) -> Node | None:
    prev = None
    while head:
        nxt = head.next
        head.next = prev
        prev = head
        head = nxt
    return prev
```

Три указателя: `prev`, `head`, `nxt`. Сохрани `next` до перезаписи, иначе потеряешь хвост. O(n) времени, O(1) памяти. Рекурсивная версия проще, но ест стек.
"""),
    E("algo", "two pointers",
      "Как работает метод двух указателей?",
      """
Указатели двигаются навстречу или в одном направлении, каждый шаг отбрасывает часть вариантов.

```python
def two_sum_sorted(a: list[int], target: int) -> tuple[int, int] | None:
    lo, hi = 0, len(a) - 1
    while lo < hi:
        s = a[lo] + a[hi]
        if s == target:
            return lo, hi
        if s < target:
            lo += 1
        else:
            hi -= 1
    return None

def max_sum_window(a: list[int], k: int) -> int:
    cur = sum(a[:k])
    best = cur
    for i in range(k, len(a)):
        cur += a[i] - a[i - k]
        best = max(best, cur)
    return best
```

Работает на отсортированных данных или на окне фиксированного размера. Сложность O(n) вместо O(n²) у перебора пар.
"""),
    E("algo", "top k heap",
      "Как найти k самых больших элементов за O(n log k)?",
      """
Минимальная куча размера k: новый элемент вытесняет минимум.

```python
import heapq

def top_k(values: list[int], k: int) -> list[int]:
    if k <= 0:
        return []
    heap: list[int] = []
    for x in values:
        if len(heap) < k:
            heapq.heappush(heap, x)
        elif x > heap[0]:
            heapq.heapreplace(heap, x)
    return sorted(heap, reverse=True)

print(top_k([5, 1, 9, 3, 7], 3))   # [9, 7, 5]
```

`heapreplace` делает pop+push за O(log k) и одним вызовом. Для всего массива проще `sorted(values, reverse=True)[:k]` — O(n log n), иногда быстрее на практике.
"""),
    E("algo", "big o",
      "Дай таблицу сложностей основных операций.",
      """
| Структура | Доступ | Поиск | Вставка | Удаление |
|---|---|---|---|---|
| Массив | O(1) | O(n) | O(n) | O(n) |
| Связный список | O(n) | O(n) | O(1)* | O(1)* |
| Хэш-таблица | — | O(1) ср. | O(1) ср. | O(1) ср. |
| Сбалансированное дерево | O(log n) | O(log n) | O(log n) | O(log n) |
| Куча | O(1) min | O(n) | O(log n) | O(log n) |

\* если есть указатель на узел.

Сортировка сравнением — не быстрее O(n log n). Вложенные циклы по n — O(n²), логарифм появляется при делении задачи вдвое.
"""),
    E("algo", "memoization",
      "Как превратить рекурсию с повторами в динамику через мемоизацию?",
      """
```python
from functools import cache

@cache
def fib(n: int) -> int:
    if n < 2:
        return n
    return fib(n - 1) + fib(n - 2)

print(fib(90))   # fast, no exponential blowup
```

`@cache` (Python 3.9) хранит результаты по аргументам; аргументы должны быть хэшируемыми. Без мемоизации `fib(40)` делает ~2·10⁸ вызовов, с ней — 40. Для больших n добавь `sys.setrecursionlimit` или перепиши итеративно.
"""),
    E("algo", "bit tricks",
      "Покажи полезные битовые трюки.",
      """
```python
def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0

def count_bits(n: int) -> int:
    count = 0
    while n:
        n &= n - 1        # clears lowest set bit
        count += 1
    return count

def lowest_set_bit(n: int) -> int:
    return n & -n

print([count_bits(i) for i in range(8)])   # [0,1,1,2,1,2,2,3]
print(lowest_set_bit(12))                   # 4
```

`n & (n-1)` убирает младший установленный бит, `n & -n` его выделяет. В Python целые беззнаковые не ограничены, маскируй `& 0xFFFFFFFF`, если нужен 32-битный результат.
"""),
)

# ---------------------------------------------------------------------------
# Tooling
# ---------------------------------------------------------------------------
add(
    E("tools", "git",
      "Дай рабочий набор git-команд для ветки и слияния.",
      """
```bash
git switch -c feature/login          # create branch
git add -A
git commit -m "add login form"
git fetch origin
git rebase origin/main               # replay on top of main
git push -u origin feature/login
```

`rebase` переписывает историю ветки — не делай его на общей ветке. Конфликт: правь файл, `git add`, `git rebase --continue`; отмена — `git rebase --abort`. Для отката опубликованного коммита — `git revert <sha>`, не `reset --hard`.
"""),
    E("tools", "gdb",
      "Покажи базовые команды gdb.",
      """
```bash
gcc -g -O0 app.c -o app
gdb ./app
```

```text
break main
run arg1 arg2
next          # step over
step          # step into
continue
print x
print *ptr
x/16gx $rsp
bt
info registers
layout src
quit
```

`-g -O0` даёт нормальные имена и переменные. Для падения — `ulimit -c unlimited`, затем `gdb ./app core` и `bt full`. `watch x` останавливает при изменении переменной.
"""),
    E("tools", "docker",
      "Собери минимальный Docker-образ для Python-приложения.",
      """
```dockerfile
FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN useradd -m appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t extra-code:latest .
docker run --rm -p 8000:8000 extra-code:latest
```

`--no-cache-dir` уменьшает слой, `USER` обязателен для продакшена. `.dockerignore` с `.venv`, `.git`, `data/raw` ускоряет сборку. Для меньшего образа — multi-stage и `python:3.12-alpine`, но многие wheels под musl не собираются.
"""),
    E("tools", "regex",
      "Покажи практические регулярные выражения и ловушки.",
      """
```python
import re

text = "user=ann ip=10.0.0.5 code=404"

m = re.search(r"\\bip=(\\d{1,3}(?:\\.\\d{1,3}){3})\\b", text)
print(m.group(1))                       # 10.0.0.5

print(re.findall(r"\\w+=(\\w+)", text)) # ['ann', '10', '404'] - see the trap

pairs = dict(re.findall(r"(\\w+)=([^\\s]+)", text))
print(pairs)                            # {'user': 'ann', 'ip': '10.0.0.5', 'code': '404'}
```

Жадные `.*` и `.+` съедают лишнее — делай ленивыми (`.*?`) или ограничивай классом. `re.match` якорит начало, `re.search` ищет везде. Экранируй точку: `\\.` — литерал, `.` — любой символ.
"""),
    E("tools", "sql",
      "Объясни JOIN и индекс на примере.",
      """
```sql
SELECT u.name, COUNT(o.id) AS orders
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
WHERE u.created_at >= '2026-01-01'
GROUP BY u.id, u.name
ORDER BY orders DESC
LIMIT 20;
```

`INNER JOIN` оставляет только совпадения, `LEFT JOIN` — всех пользователей, у кого нет заказов, `COUNT(o.id)` даст 0. Индекс:

```sql
CREATE INDEX idx_orders_user_created ON orders (user_id, created_at);
```

Порядок колонок в составном индексе важен: сначала равенство, потом диапазон. `EXPLAIN ANALYZE` показывает, использует ли планировщик индекс.
"""),
    E("tools", "http",
      "Как отправить JSON-запрос через curl и проверить ответ?",
      """
```bash
curl -sS -X POST https://api.example.com/v1/chat \\
  -H "Authorization: Bearer $TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{"model":"m","messages":[{"role":"user","content":"hi"}]}' \\
  -w '\\nHTTP %{http_code} in %{time_total}s\\n' \\
  -o response.json

python -c "import json;print(json.load(open('response.json'))['choices'][0]['message']['content'])"
```

`-sS` — тихо, но с ошибками; `-w` печатает код и время; `-o` пишет тело в файл. Не передавай токен в URL — он попадёт в логи прокси.
"""),
    E("tools", "ssh",
      "Как настроить ключ SSH и подключиться без пароля?",
      """
```bash
ssh-keygen -t ed25519 -C "sam@host" -f ~/.ssh/id_ed25519
ssh-copy-id -i ~/.ssh/id_ed25519.pub user@server
ssh user@server
```

```text
# ~/.ssh/config
Host vps
    HostName 203.0.113.10
    User root
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 30
```

Права: `~/.ssh` — 700, ключи — 600, иначе `Permission denied (publickey)`. Приватный ключ не копируется на сервер и не коммитится.
"""),
    E("tools", "strace",
      "Как найти, какие файлы открывает программа?",
      """
```bash
strace -f -e trace=openat,read,write,connect ./app 2>&1 | head -50
strace -f -c ./app                 # syscall summary
ltrace ./app                       # library calls
```

На Windows аналог — Process Monitor (`Procmon`) с фильтром по `Process Name`. `strace -p PID` цепляется к живому процессу. `-f` важен для многопоточных и форкающихся программ.
"""),
    E("tools", "wireshark",
      "Дай полезные фильтры Wireshark.",
      """
```text
http.request.method == "POST"
tls.handshake.type == 1
ip.addr == 192.168.1.10 && tcp.port == 443
dns.qry.name contains "example.com"
tcp.flags.syn == 1 && tcp.flags.ack == 0
```

`tshark -r capture.pcap -Y 'http.request' -T fields -e http.host -e http.request.uri` — то же из консоли. TLS без ключей не расшифровать; для своего трафика задай `SSLKEYLOGFILE` и укажи файл в настройках TLS.
"""),
    E("tools", "venv",
      "Как поднять виртуальное окружение и зафиксировать зависимости?",
      """
```bash
python -m venv .venv
.venv\\Scripts\\activate           # Windows
# source .venv/bin/activate      # Linux/macOS
python -m pip install -U pip
pip install fastapi uvicorn httpx
pip freeze > requirements.txt
deactivate
```

`pip freeze` пишет точные версии вместе с транзитивными; для библиотеки лучше `pyproject.toml` с диапазонами. Не ставь пакеты глобально — конфликт версий обеспечен.
"""),
)


# ---------------------------------------------------------------------------
# Cross-language comparisons
# ---------------------------------------------------------------------------
add(
    E("compare", "memory management",
      "Сравни управление памятью в C, C++, Java, C# и Python.",
      """
- C: `malloc/free`, ошибка — утечка или UB, ответственность на программисте.
- C++: RAII и умные указатели; `new/delete` почти не нужны в прикладном коде.
- Java: сборщик мусора, ссылки; утечки возможны через забытые ссылки (кэши, слушатели).
- C#: GC + `IDisposable` для внешних ресурсов, `using` детерминированно закрывает.
- Python: подсчёт ссылок + циклический GC; `with` для файлов и соединений.

Правило: владение должно быть очевидным. Если непонятно, кто освобождает — это баг, даже если он пока не стреляет.
"""),
    E("compare", "generics",
      "Чем generics в Java отличаются от C# и шаблонов C++?",
      """
- C++: мономорфизация, отдельный код на каждый тип, ошибки на этапе компиляции, нет общего базового типа.
- C#: JIT создаёт код для value type, reference type делят один; ограничения проверяются компилятором.
- Java: стирание до bound, проверка в рантайме частичная; нельзя `new T()` и `T[]`.

C++ даёт максимальную скорость и размер кода, Java — совместимость, C# — баланс.
"""),
    E("compare", "exceptions",
      "Как устроены исключения в C++, Java, C# и Python?",
      """
- C++: исключения по значению, ловля по `const&`, раскрутка стека вызывает деструкторы; поддерживает `noexcept` для оптимизаций.
- Java: checked (`IOException`) и unchecked (`RuntimeException`), `finally` выполняется всегда.
- C#: только unchecked, `using`/`try-finally`, фильтры `catch (Ex e) when (...)`.
- Python: всё исключения, `finally`, `raise ... from`, контекстные менеджеры вместо RAII.

Общее: не глотай ошибку без записи, не используй исключения для обычного потока управления.
"""),
    E("compare", "strings",
      "Как отличаются строки в Python, C++, Java, C# и C?",
      """
- C: `char*` + длина отдельно, нужен `\\0`, ручное управление.
- C++: `std::string` (владеет), `std::string_view` (срез), small-string optimization.
- Java: `String` неизменяемый, `StringBuilder` для сборки, интернирование литералов.
- C#: `string` неизменяемый, `Span<char>`/`StringBuilder` для работы без аллокаций.
- Python: `str` неизменяемый Unicode, срезы копируют, `bytearray` для байтов.

Неизменяемость даёт безопасные хэш-ключи и потоки, но конкатенация в цикле — O(n²); используй builder/join.
"""),
)

# ---------------------------------------------------------------------------
# Extra distillation prompts (questions only)
# ---------------------------------------------------------------------------
EXTRA_QUESTIONS: list[tuple[str, str]] = [
    ("python", "Как работает механизм импорта Python и что такое __init__.py?"),
    ("python", "Что такое дескрипторы и как они используются в свойствах?"),
    ("python", "Как работает сборка мусора и подсчёт ссылок в CPython?"),
    ("python", "Что такое метаклассы и когда они реально нужны?"),
    ("python", "Как профилировать Python-код: cProfile, py-spy, line_profiler?"),
    ("python", "Как написать асинхронный TCP-сервер на asyncio?"),
    ("python", "Что такое weakref и где он применяется?"),
    ("python", "Как упаковать Python-проект в wheel и опубликовать?"),
    ("python", "Как работают f-строки и спецификаторы формата?"),
    ("python", "Как безопасно работать с многопоточностью в Python?"),
    ("python", "Как устроен GIL в Python 3.13 и что такое free-threaded сборка?"),
    ("python", "Как тестировать asyncio-код через pytest-asyncio?"),
    ("cpp", "Что такое NRVO и copy elision в C++?"),
    ("cpp", "Как работают виртуальные функции и vtable?"),
    ("cpp", "Что такое perfect forwarding и универсальные ссылки?"),
    ("cpp", "Как устроены аллокаторы STL и когда писать свой?"),
    ("cpp", "Что такое constexpr и compile-time вычисления в C++20?"),
    ("cpp", "Как работать с std::optional, std::variant и std::any?"),
    ("cpp", "Что такое data race и memory_order в std::atomic?"),
    ("cpp", "Как организовать сборку через CMake с тестами и санитайзерами?"),
    ("cpp", "Что такое ABI и почему ломается совместимость библиотек?"),
    ("cpp", "Как читать ошибки шаблонов и что такое SFINAE?"),
    ("cpp", "Что нового в C++23: expected, print, mdspan?"),
    ("java", "Как работает загрузчик классов и classpath?"),
    ("java", "Что такое JVM, JRE и JDK — в чём разница?"),
    ("java", "Как работают аннотации и рефлексия в Java?"),
    ("java", "Что такое виртуальные потоки (Project Loom)?"),
    ("java", "Как устроен CompletableFuture и цепочки thenApply/thenCompose?"),
    ("java", "Как работает Spring Boot: DI, бины, автоконфигурация?"),
    ("java", "Как писать многопоточный код без гонок: synchronized, volatile, locks?"),
    ("java", "Что такое JIT-компиляция и как её наблюдать?"),
    ("java", "Как устроены коллекции ConcurrentHashMap и CopyOnWriteArrayList?"),
    ("java", "Как настроить логирование через SLF4J и Logback?"),
    ("c", "Как устроен вызов функции на уровне стека?"),
    ("c", "Что такое restrict и зачем он нужен?"),
    ("c", "Как работают выравнивание структур и padding?"),
    ("c", "Что такое volatile и почему он не заменяет атомики?"),
    ("c", "Как реализовать динамический массив (vector) на C?"),
    ("c", "Как работать с сокетами Беркли: socket, bind, listen, accept?"),
    ("c", "Как собрать проект на Makefile с несколькими целями?"),
    ("c", "Что такое endianness и как её определить в рантайме?"),
    ("c", "Как отлаживать утечки памяти через Valgrind и ASan?"),
    ("c", "Как работает препроцессор и что делают #include, #define, #if?"),
    ("csharp", "Как работает сборка мусора в .NET и поколения объектов?"),
    ("csharp", "Что такое async state machine и как компилятор переписывает await?"),
    ("csharp", "Как устроены expression trees и где они применяются?"),
    ("csharp", "Что такое dependency injection в ASP.NET Core?"),
    ("csharp", "Как работают source generators в C#?"),
    ("csharp", "Что такое ref struct и зачем ограничения на стек?"),
    ("csharp", "Как писать unit-тесты на xUnit с моками?"),
    ("csharp", "Что такое value type и reference type, boxing?"),
    ("csharp", "Как устроен LINQ to SQL и отложенное выполнение?"),
    ("csharp", "Как опубликовать NuGet-пакет?"),
    ("asm", "Как устроен стек вызовов и red zone в System V AMD64?"),
    ("asm", "Что такое PLT и GOT в ELF?"),
    ("asm", "Как написать шелл-код для Linux x86-64?"),
    ("asm", "Как работают инструкции SSE и AVX, чем отличаются от x87?"),
    ("asm", "Что такое TSC и как измерить такты?"),
    ("asm", "Как читать AT&T синтаксис и перевести в Intel?"),
    ("asm", "Что такое relocations и как линкер правит адреса?"),
    ("asm", "Как написать обработчик прерывания или syscall-хук?"),
    ("asm", "Что делает инструкция cpuid и как определить поддержку AVX2?"),
    ("asm", "Как устроен вызов виртуальной функции на ассемблере?"),
    ("re", "Как определить, что бинарник упакован или обфусцирован?"),
    ("re", "Как восстановить IAT после распаковки дампа?"),
    ("re", "Что такое RTTI и как по нему найти имена классов?"),
    ("re", "Как анализировать драйвер Windows и IOCTL-интерфейс?"),
    ("re", "Как работают хуки в kernelmode и что такое SSDT?"),
    ("re", "Как реверсить сетевой протокол по трафику?"),
    ("re", "Что такое control flow flattening и как его упростить?"),
    ("re", "Как найти ключ шифрования в бинарнике через отладчик?"),
    ("re", "Как анализировать Android APK: jadx, apktool, Frida?"),
    ("re", "Как писать скрипты для IDA на Python (IDAPython)?"),
    ("re", "Как работают hardware breakpoints и чем отличаются от software?"),
    ("re", "Что такое PEB и TEB и что там ищут анти-чит и малварь?"),
    ("algo", "Как найти цикл в графе алгоритмом Тарьяна?"),
    ("algo", "Как работает алгоритм Дейкстры и когда он ломается на отрицательных весах?"),
    ("algo", "Что такое префиксные суммы и где они ускоряют запросы?"),
    ("algo", "Как устроено дерево отрезков и lazy propagation?"),
    ("algo", "Как найти наибольшую общую подпоследовательность?"),
    ("algo", "Как работает сортировка слиянием и почему она стабильна?"),
    ("algo", "Что такое амортизированная сложность?"),
    ("algo", "Как решать задачи на бинарный поиск по ответу?"),
    ("algo", "Что такое trie и когда он лучше хэш-таблицы?"),
    ("algo", "Как устроен алгоритм KMP для поиска подстроки?"),
    ("tools", "Как читать чужой код в большом проекте?"),
    ("tools", "Как настроить CI на GitHub Actions для Python-проекта?"),
    ("tools", "Как работать с git rebase interactive и squash?"),
    ("tools", "Как отлаживать падение в продакшене по core dump?"),
    ("tools", "Как писать хорошие коммиты и код-ревью?"),
    ("tools", "Как устроен HTTP/2 и чем отличается от HTTP/1.1?"),
    ("tools", "Как работает TLS-рукопожатие?"),
    ("tools", "Как настроить nginx как reverse proxy для FastAPI?"),
    ("tools", "Как писать Dockerfile без лишних слоёв и секретов?"),
    ("tools", "Как профилировать память и CPU в Linux: perf, heaptrack?"),
]


def build_rows() -> list[dict]:
    rows = []
    for entry in TOPICS:
        rows.append({
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": entry["question"]},
                {"role": "assistant", "content": entry["answer"]},
            ],
            "task": "code",
            "meta": {"lang": entry["lang"], "topic": entry["topic"], "level": entry["level"]},
        })
    return rows


def build_prompts() -> list[dict]:
    rows = [
        {"messages": [{"role": "system", "content": SYSTEM},
                      {"role": "user", "content": question}]}
        for _, question in EXTRA_QUESTIONS
    ]
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the code-teaching SFT set")
    parser.add_argument("--out", default="data/sft")
    args = parser.parse_args()
    out = Path(args.out)

    rows = build_rows()
    prompts = build_prompts()
    write_jsonl(out / "code_seed.jsonl", rows)
    write_jsonl(out / "code_prompts.jsonl", prompts)

    by_lang: dict[str, int] = {}
    for row in rows:
        lang = row["meta"]["lang"]
        by_lang[lang] = by_lang.get(lang, 0) + 1
    print(json.dumps({
        "code_rows": len(rows),
        "distill_prompts": len(prompts),
        "by_lang": by_lang,
        "avg_answer_chars": round(sum(len(r["messages"][2]["content"]) for r in rows) / max(1, len(rows))),
        "out": str(out),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
