import argparse
import json
import random
import time
import urllib.parse
import urllib.request


QUERIES = [
    "machine learning algorithms",
    "climate change effects",
    "data privacy protection",
    "electric vehicle charging",
    "home gardening vegetables",
    "online learning platforms",
    "quantum computing principles",
    "sleep quality improvement",
    "smartphone photography tips",
    "space exploration missions",
    "sustainable fashion brands",
    "travel budget planning",
    "artificial intelligence ethics",
    "cooking pasta recipes",
    "meditation stress relief",
    "tesla battery technology",
    "investment strategies",
    "exercise weight loss",
    "how to use computer",
    "kamen rider",
    "web search ranking",
    "natural language processing",
    "information retrieval models",
    "student study methods",
    "healthy breakfast ideas",
    "renewable energy storage",
    "python programming tutorial",
    "database indexing basics",
    "deep learning applications",
    "user privacy online",
    "search engine evaluation",
]


def post_json(url, payload, timeout=10):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout).read()


def run(base_url, seed):
    modes = ["adaptive"] * 25 + ["baseline"] * 16 + ["ltr"] * 16 + ["cross_encoder"] * 8
    rng = random.Random(seed)
    rng.shuffle(modes)

    created = []
    errors = []
    for idx, mode in enumerate(modes, start=1):
        session = f"demo_human_20260412_{idx:03d}"
        query = rng.choice(QUERIES)
        params = {
            "q": query,
            "mode": mode,
            "session_id": session,
            "hl": "true",
        }
        if mode in {"ltr", "cross_encoder"}:
            params["rerank_top_n"] = "10"

        url = f"{base_url}/search?{urllib.parse.urlencode(params)}"
        started = time.perf_counter()
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = json.loads(response.read().decode("utf-8"))
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            results = payload.get("results") or []
            search_id = payload.get("search_id")
            created.append((mode, query, len(results), round(elapsed_ms, 1)))

            # Light click simulation; do not duplicate query_submitted events.
            if results and rng.random() < 0.55:
                rank = rng.randint(1, min(3, len(results)))
                clicked = results[rank - 1]
                post_json(
                    f"{base_url}/log",
                    {
                        "type": "result_clicked",
                        "sessionId": session,
                        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                        "query": query,
                        "searchId": search_id,
                        "docId": clicked.get("_id"),
                        "rank": rank,
                        "source": "simulated_demo_human",
                    },
                )
        except Exception as exc:
            errors.append((mode, query, repr(exc)))
        time.sleep(rng.uniform(0.03, 0.12))

    return created, errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:5000")
    parser.add_argument("--seed", type=int, default=2254400)
    args = parser.parse_args()

    created, errors = run(args.base_url.rstrip("/"), args.seed)
    counts = {}
    for mode, *_ in created:
        counts[mode] = counts.get(mode, 0) + 1

    print(f"created_searches={len(created)}")
    print(f"mode_counts={counts}")
    print(f"errors={len(errors)}")
    if errors:
        print(f"first_errors={errors[:5]}")
    print("last_10=")
    for item in created[-10:]:
        print(item)


if __name__ == "__main__":
    main()
