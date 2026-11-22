"""Deterministic output polish: things a 3B keeps getting wrong.

Measured defects on this box, each one fixed after generation so every
door (API, console, agent) gets the same text:

  * the model prefixes answers with its own name ("Extra Code v1") — a
    prompt rule alone did not stop it;
  * Python arrives with every body line at the SAME indent, i.e. all
    nesting flattened to one level. Invalid Python, unusable as copied.
    Scaling the indent cannot fix it (there is no relative signal left),
    so the block structure is rebuilt from the `:` stack;
  * CJK runs get glued into Russian words ("нам看不见") — a training
    leak, not content. This deployment answers ru/en.
"""
from __future__ import annotations

import re

_NAME_LINE = re.compile(
    r"^\s*(?:я\s*[—–-]\s*)?extra\s*code(?:\s*v1)?\s*[.:!]?\s*$",
    re.IGNORECASE,
)
_PY_FENCE = re.compile(r"```(?:python|py)\n(.*?)```", re.S)
# CJK glued to Cyrillic is a training leak, not content.
_CJK_LEAK = re.compile(
    r"(?<=[а-яА-ЯёЁ])[\u3000-\u303f\u3400-\u9fff]+"
    r"|[\u3000-\u303f\u3400-\u9fff]+(?=[а-яА-ЯёЁ])"
)

# Module-level statements that reset the stack: in flattened dumps they
# still mark the end of the previous suite. Any other flat line keeps the
# current depth — measured: resetting on `if` after `def` produced
# "expected an indented block" and shipped broken code.
_TOPLEVEL = ("def ", "async def ", "class ", "@", "import ", "from ")

# Keywords that continue a block at the SAME depth instead of opening a
# new one: the body below them is the sibling branch.
_BRANCH = (
    "elif ", "else:", "except", "finally:", "case ", "case:", "default:",
)
# Lines that legitimately end a suite and start a new top-level statement.
_DEF = ("def ", "async def ", "class ", "@")


def strip_name_prefix(text: str) -> str:
    """Drop a leading line that is just the model name.

    Never returns empty: when the whole answer is the name (a legitimate
    reply to "кто ты"), the original text is kept. Measured bug: the
    identity answer got erased to "" by an unconditional strip.
    """
    lines = text.splitlines()
    changed = False
    while lines and not lines[0].strip():
        lines.pop(0)
        changed = True
    if lines and _NAME_LINE.match(lines[0]):
        rest = lines[1:]
        while rest and not rest[0].strip():
            rest.pop(0)
        if rest:
            lines = rest
            changed = True
        # no rest -> the name IS the answer, leave it alone
    out = "\n".join(lines)
    return out if (changed and out.strip()) else text


def strip_cjk_leak(text: str) -> str:
    """Remove CJK runs welded onto Cyrillic words."""
    return _CJK_LEAK.sub("", text)


def _code_only(line: str) -> str:
    """Line with strings and trailing comment removed, for structure tests."""
    out: list[str] = []
    quote = ""
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if ch == quote:
                quote = ""
            i += 1
            continue
        if ch in "\"'":
            quote = ch
            i += 1
            continue
        if ch == "#":
            break
        out.append(ch)
        i += 1
    return "".join(out).rstrip()


def _opens_block(line: str) -> bool:
    """True when this line must be followed by a deeper suite."""
    code = _code_only(line)
    if not code.endswith(":"):
        return False
    # dict/slice/lambda annotations and type hints end with ':' too
    if code.endswith("::"):
        return False
    head = code.lstrip()
    if head.startswith(("lambda", "#")):
        return False
    return True


def _is_branch(line: str) -> bool:
    return line.lstrip().startswith(_BRANCH)


def _compiles(block: str) -> bool:
    try:
        compile(block, "<polish>", "exec")
    except (SyntaxError, ValueError):
        return False
    return True


def _is_flat(block: str) -> bool:
    """True when nesting information is already gone from the text.

    The rebuild below infers structure from `:` openers alone, which is
    only sound when the model emitted every line at one indent. Two shapes
    count as flat: everything at one column, and a header line at column 0
    with the whole suite uniformly shifted (the 4B writes one space per
    line under a long system prompt — relative structure is gone).
    Anything with real varying indents is returned untouched: replacing
    actual structure with a guess is worse than shipping it broken.
    """
    lines = [raw for raw in block.splitlines() if raw.strip()]
    if not lines:
        return False
    indents = {len(raw) - len(raw.lstrip()) for raw in lines}
    if len(indents) <= 1:
        return True
    body_indents = {len(raw) - len(raw.lstrip()) for raw in lines[1:]}
    return len(indents) <= 2 and len(body_indents) <= 1


def rebuild_python(block: str) -> str:
    """Rebuild suite nesting from `:` openers.

    Used only when the block is flat: the common failure is a whole body
    at one indent level, where the original indent carries no nesting
    information.

    Rules, tuned on real model output (guard-style snippets):
      * `def`/`class` open a suite that stays open;
      * a control-flow opener (if/for/while/with/try) owns ONE statement
        — the next line is read as back at the enclosing level. Guards
        and single-statement loops are what flattened snippets almost
        always are, and this reading compiles with plausible structure;
      * a branch keyword (else/except/finally/elif) lands at its
        sibling level and owns one statement too.

    Top-level lines (original indent 0) reset the stack, so a trailing
    `raise` after the function still lands at column 0.
    """
    lines = block.splitlines()
    out: list[str] = []
    depth = 0
    quota = 0
    # level where the sibling branch of the last opener sits (elif/else
    # return here; the one-statement quota may have closed the suite
    # already, so a plain depth-1 would overshoot)
    suite_parent = 0
    in_string = ""
    for raw in lines:
        if not raw.strip():
            out.append("")
            continue
        stripped = raw.strip()
        indent = len(raw) - len(raw.lstrip())
        # Inside a triple-quoted string the content is verbatim.
        if in_string:
            out.append("    " * depth + stripped if indent else stripped)
            if in_string in stripped:
                in_string = ""
            continue
        for token in ('"""', "'''"):
            if stripped.count(token) % 2 == 1:
                in_string = token
                break
        if indent == 0 and depth > 0 and stripped.startswith(_TOPLEVEL) \
                and not _is_branch(stripped):
            depth = 0
            quota = 0
            suite_parent = 0
        is_def = stripped.startswith(("def ", "async def ", "class ", "@"))
        # A branch keyword sits next to its opener, not inside the suite.
        if _is_branch(stripped):
            depth = suite_parent
            quota = 0
            out.append("    " * depth + stripped)
            if _opens_block(stripped):
                depth += 1
                quota = 1
                # suite_parent stays: the next elif/else lands there too
            continue
        if quota:
            # the one statement of the suite: emit it at suite depth,
            # then the stream returns to the enclosing level
            out.append("    " * depth + stripped)
            quota = 0
            if _opens_block(stripped):
                suite_parent = depth
                depth += 1
                quota = 1 if not is_def else 0
            else:
                depth = max(0, depth - 1)
            continue
        out.append("    " * depth + stripped)
        if _opens_block(stripped):
            suite_parent = depth
            depth += 1
            quota = 0 if is_def else 1
    return "\n".join(out)


def fix_python_indent(text: str) -> str:
    """Repair a Python fence only when the code does not compile AND the
    block is flat. Anything else is returned untouched: indentation that
    still carries relative structure is the model's own and must not be
    overwritten by a heuristic."""

    def repl(match: re.Match[str]) -> str:
        block = match.group(1)
        # The closing fence may be welded onto the last statement; the
        # fence is presentation, so it always starts on its own line.
        body = block if block.endswith("\n") else block + "\n"
        if _compiles(block) or not _is_flat(block):
            return "```python\n" + body + "```"
        fixed = rebuild_python(block)
        if not _compiles(fixed):
            return "```python\n" + body + "```"
        return "```python\n" + fixed.rstrip("\n") + "\n```"

    return _PY_FENCE.sub(repl, text)


def polish(text: str) -> str:
    if not text:
        return text
    return fix_python_indent(strip_cjk_leak(strip_name_prefix(text)))
