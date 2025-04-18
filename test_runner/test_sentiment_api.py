import os
import time
import json
import csv
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("BASE_URL")
RUNS = 100

ENDPOINTS = {
    "/get-sentiment-ultra-basic": "../tests/data/shortform.json",
    "/get-sentiment-ultra-sections": "../tests/data/ultrasections.json",
    "/get-sentiment-long-form": "../tests/data/longform.json"
}

def load_data(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def measure(endpoint, data):
    total = 0.0
    for _ in range(RUNS):
        start = time.time()
        requests.post(BASE_URL + endpoint, json=data)
        total += time.time() - start
    return total / RUNS

def get_next_csv():
    i = 1
    while os.path.exists(f"test{i}.csv"):
        i += 1
    return f"test{i}.csv"

def main():
    results = []
    for ep, path in ENDPOINTS.items():
        data = load_data(path)
        avg = round(measure(ep, data), 4)
        results.append({
            "endpoint": ep,
            "numberOfRuns": RUNS,
            "avgTime": avg
        })
    file = get_next_csv()
    with open(file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["endpoint", "numberOfRuns", "avgTime"])
        writer.writeheader()
        writer.writerows(results)
    print(f"✅ Results saved to {file}")

if __name__ == "__main__":
    main()
