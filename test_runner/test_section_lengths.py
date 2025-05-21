#!/usr/bin/env python3
# section_latency_probe_x100.py
import os
import time
import json
import csv
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
RUNS = 100                          # one hundred posts per file
ENDPOINT = "/get-sentiment-ultra-sections"

SECTION_FILES = [
    "../tests/data/sectionLengths/section10.json",
    "../tests/data/sectionLengths/section20.json",
    "../tests/data/sectionLengths/section30.json",
    "../tests/data/sectionLengths/section40.json",
    "../tests/data/sectionLengths/section50.json",
]

# --------------------------------------------------------------------------- #


def load_json(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def time_single_call(payload: dict) -> float:
    start = time.perf_counter()
    requests.post(BASE_URL + ENDPOINT, json=payload)
    return time.perf_counter() - start


def next_csv_name() -> str:
    i = 1
    while os.path.exists(f"section_tests{i}_x100.csv"):
        i += 1
    return f"section_tests{i}_x100.csv"

# --------------------------------------------------------------------------- #


def main():
    rows = []
    for path in SECTION_FILES:
        data = load_json(path)
        n_sections = len(data.get("sections", []))

        for run in range(1, RUNS + 1):
            elapsed = round(time_single_call(data), 4)
            rows.append(
                {
                    "sections": n_sections,
                    "run": run,
                    "elapsed_seconds": elapsed,
                }
            )

    outfile = next_csv_name()
    with open(outfile, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["sections", "run", "elapsed_seconds"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} measurements to {outfile}")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
