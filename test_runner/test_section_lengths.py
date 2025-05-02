import os
import time
import json
import csv
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
RUNS = 100  # number of POSTs per file
ENDPOINT = "/get-sentiment-ultra-sections"

SECTION_FILES = [
    "../tests/data/sectionLengths/section10.json",
    "../tests/data/sectionLengths/section20.json",
    "../tests/data/sectionLengths/section30.json",
    "../tests/data/sectionLengths/section40.json",
    "../tests/data/sectionLengths/section50.json",
]


def load_json(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def avg_latency(data: dict) -> float:
    total = 0.0
    for _ in range(RUNS):
        start = time.time()
        requests.post(BASE_URL + ENDPOINT, json=data)
        total += time.time() - start
    return total / RUNS


def next_csv_name() -> str:
    i = 1
    while os.path.exists(f"section_tests{i}.csv"):
        i += 1
    return f"section_tests{i}.csv"


def main():
    results = []

    for path in SECTION_FILES:
        payload = load_json(path)
        num_sections = len(payload.get("sections", []))
        avg_time = round(avg_latency(payload), 4)

        results.append(
            {
                "number of sections": num_sections,
                "number of runs": RUNS,
                "average time": avg_time,
            }
        )

    csv_name = next_csv_name()
    with open(csv_name, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["number of sections", "number of runs", "average time"],
        )
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {csv_name}")


if __name__ == "__main__":
    main()
