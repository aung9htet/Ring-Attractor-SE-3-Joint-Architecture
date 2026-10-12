#!/usr/bin/env python3
"""Fail when a summary quotes a number that is not in the generated tables.

    python3 -B scripts/check_doc_numbers.py

Checked: every number with a decimal point or a percent sign in the summary
sections of README.md (section 1) and docs/results.md (the summary and the
claims table).  Each must occur in docs/results/tables.md, which
scripts/report.py writes from the data.  The same for the inverse kinematics:
README.md (section 11) and docs/ik/results.md (the summary and the claims
table) against docs/ik/results/tables.md, which scripts/ik_report.py writes.
Constants of the method that are not results (pass marks, tick length,
thresholds) are listed in ALLOWED.
"""

import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
TABLES = os.path.join("docs", "results", "tables.md")
IK_TABLES = os.path.join("docs", "ik", "results", "tables.md")
#: (file, start marker, end marker, tables): the text between the markers is checked against the tables
SECTIONS = (
    ("README.md", "## 1. ", "## 2. ", TABLES),
    (os.path.join("docs", "results.md"), "## Summary", "## 1. ", TABLES),
    (os.path.join("docs", "results.md"), "## 12. The claims", "## 13. ", TABLES),
    ("README.md", "## 11. ", "## 12. ", IK_TABLES),
    (os.path.join("docs", "ik", "results.md"), "## Summary", "## 1. ", IK_TABLES),
    (os.path.join("docs", "ik", "results.md"), "## 10. The claims", "## 11. ", IK_TABLES),
)
#: not results: pass marks, settings and the chance level
ALLOWED = {"0.5", "0.3", "0.05", "0.02", "0.03", "1.0", "0.25", "126.5", "1.8", "0.7", "70%", "1.5", "0.1"}
#: section numbers of README.md, section 11
ALLOWED |= {"11.1", "11.2", "11.3", "11.4"}
NUMBER = re.compile(r"(?<![\w.])\d[\d,]*\.\d+%?|(?<![\w.])\d+%")


def section(text, start, end):
    if start not in text:
        return None
    body = text[text.index(start):]
    return body[: body.index(end)] if end in body else body


def numbers(text):
    # code spans and links are not prose
    text = re.sub(r"`[^`]*`", " ", text)
    text = re.sub(r"\]\([^)]*\)", "]", text)
    return sorted({match.group(0).replace(",", "") for match in NUMBER.finditer(text)})


def check(root=ROOT):
    """Return ``(checked, missing)``; ``missing`` is a list of ``(file, number)``.  A section whose tables are not
    generated yet is left out."""

    checked, missing = check_against(root)
    return checked, [(name, number) for name, number, _ in missing]


def check_against(root=ROOT):
    """As :func:`check`, with the tables each missing number was looked for in: ``(file, number, tables)``."""

    checked, missing, read = 0, [], {}
    for name, start, end, source in SECTIONS:
        path = os.path.join(root, name)
        if not os.path.exists(path) or not os.path.exists(os.path.join(root, source)):
            continue
        if source not in read:
            with open(os.path.join(root, source), encoding="utf-8") as stream:
                text = stream.read().replace(",", "")
            read[source] = (text, set(NUMBER.findall(text)))
        tables, known = read[source]
        with open(path, encoding="utf-8") as stream:
            body = section(stream.read(), start, end)
        if body is None:
            continue
        for number in numbers(body):
            checked += 1
            bare = number.rstrip("%")
            # a percentage may stand in a table as a bare number under a heading that says "%"
            whole = bare.isdigit() and re.search(r"(?<![\w.])%s(?![\w.%%])" % bare, tables) is not None
            if number not in known and bare not in known and not whole and number not in ALLOWED:
                missing.append((name, number, source))
    return checked, missing


def main():
    checked, missing = check_against()
    for name, number, source in missing:
        print("%s quotes %s, which is not in %s" % (name, number, source))
    print("%d numbers checked, %d not found" % (checked, len(missing)))
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
