import os
import time
import json
import csv
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
RUNS = 100  # number of runs per endpoint

ENDPOINTS = {
    "/get-sentiment-ultra-basic":   "../tests/data/shortform.json",
    "/get-sentiment-ultra-sections": "../tests/data/ultrasections.json",
    "/get-sentiment-long-form":    "../tests/data/longform.json",
}

# ---------- helpers ---------------------------------------------------------


def load_data(path: str):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def time_single_call(endpoint: str, payload: dict) -> float:
    """POST once and return elapsed time in seconds."""
    start = time.perf_counter()
    requests.post(BASE_URL + endpoint, json=payload)
    return time.perf_counter() - start


def next_csv_name() -> str:
    i = 1
    while os.path.exists(f"test{i}_x100.csv"):
        i += 1
    return f"test{i}_x100.csv"

# ---------- main ------------------------------------------------------------


def main():
    rows = []  # one row per run
    for ep, path in ENDPOINTS.items():
        data = load_data(path)
        for run_idx in range(1, RUNS + 1):
            elapsed = round(time_single_call(ep, data), 4)
            rows.append(
                {
                    "endpoint": ep,
                    "run": run_idx,
                    "elapsed_seconds": elapsed,
                }
            )

    outfile = next_csv_name()
    with open(outfile, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["endpoint", "run", "elapsed_seconds"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {len(rows)} measurements to {outfile}")


if __name__ == "__main__":
    main()
