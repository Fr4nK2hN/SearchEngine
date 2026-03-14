import time

from elasticsearch import Elasticsearch

from index_setup import INDEX_NAME, get_current_index_meta, get_expected_index_spec, index_meta_matches


ES_CLIENT = Elasticsearch([{"host": "elasticsearch", "port": 9200, "scheme": "http"}])


def wait_for_index(timeout=300, interval=2):
    _, _, expected_meta = get_expected_index_spec()
    deadline = time.time() + timeout
    last_error = None

    while time.time() < deadline:
        try:
            if not ES_CLIENT.ping():
                raise RuntimeError("Elasticsearch is not reachable yet")

            if not ES_CLIENT.indices.exists(index=INDEX_NAME):
                raise RuntimeError(f"Index '{INDEX_NAME}' does not exist yet")

            count = ES_CLIENT.count(index=INDEX_NAME)["count"]
            current_meta = get_current_index_meta(ES_CLIENT, INDEX_NAME)
            if count > 0 and index_meta_matches(current_meta, expected_meta):
                print(
                    f"✓ Index '{INDEX_NAME}' is ready with fingerprint "
                    f"{expected_meta['index_fingerprint'][:12]} and {count} docs."
                )
                return

            if count <= 0:
                last_error = RuntimeError(f"Index '{INDEX_NAME}' exists but is empty")
            else:
                last_error = RuntimeError(
                    f"Index '{INDEX_NAME}' fingerprint mismatch "
                    f"(current={current_meta.get('index_fingerprint')}, "
                    f"expected={expected_meta['index_fingerprint']})"
                )
        except Exception as exc:
            last_error = exc

        print(f"Waiting for ready index '{INDEX_NAME}': {last_error}")
        time.sleep(interval)

    raise RuntimeError(
        f"Timed out waiting for index '{INDEX_NAME}' to become ready: {last_error}"
    )


if __name__ == "__main__":
    wait_for_index()
