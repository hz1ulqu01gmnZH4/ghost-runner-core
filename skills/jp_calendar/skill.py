"""jp_calendar: Gregorian date -> Japanese era (和暦) and weekday.

stdin:  {"date": "YYYY-MM-DD"}
stdout: {"date", "era", "era_year", "wareki", "weekday"}
"""
import datetime
import json
import sys

ERAS = [  # (first day, era name) — newest first
    (datetime.date(2019, 5, 1), "令和"),
    (datetime.date(1989, 1, 8), "平成"),
    (datetime.date(1926, 12, 25), "昭和"),
    (datetime.date(1912, 7, 30), "大正"),
    (datetime.date(1868, 1, 25), "明治"),
]
WEEKDAYS = ["月曜日", "火曜日", "水曜日", "木曜日", "金曜日", "土曜日", "日曜日"]


def main() -> None:
    args = json.load(sys.stdin)
    raw = args.get("date")
    if not isinstance(raw, str):
        raise ValueError("args.date must be a YYYY-MM-DD string")
    date = datetime.date.fromisoformat(raw)
    for start, era in ERAS:
        if date >= start:
            era_year = date.year - start.year + 1
            break
    else:
        raise ValueError(f"{raw} is before 明治 (1868-01-25); no era mapping")
    year_label = "元年" if era_year == 1 else f"{era_year}年"
    print(json.dumps({
        "date": raw,
        "era": era,
        "era_year": era_year,
        "wareki": f"{era}{year_label}{date.month}月{date.day}日",
        "weekday": WEEKDAYS[date.weekday()],
    }, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # one skill contract: any failure = stderr + exit 1
        print(f"jp_calendar: {exc}", file=sys.stderr)
        sys.exit(1)
