import fcntl
import os
import time
from contextlib import contextmanager

from elasticsearch.helpers import bulk

from engine.index_setup import (
    INDEX_NAME,
    build_source_document,
    get_current_index_meta,
    get_expected_index_spec,
    index_meta_matches,
    load_documents,
)
from ..bootstrap import create_es_client
from ..config import load_config


RUNTIME_CONFIG = load_config()
ES_CLIENT = create_es_client(RUNTIME_CONFIG)
INDEX_BUILD_LOCK_PATH = os.getenv(
    "INDEX_BUILD_LOCK_PATH",
    "/tmp/searchengine-documents-index.lock",
)


def wait_for_elasticsearch(timeout=180, interval=2):
    deadline = time.time() + timeout
    last_error = None
    while time.time() < deadline:
        try:
            if ES_CLIENT.ping():
                print("✓ Elasticsearch is reachable.")
                return
            last_error = RuntimeError("Elasticsearch ping returned false")
        except Exception as exc:
            last_error = exc
        print(f"Waiting for Elasticsearch: {last_error}")
        time.sleep(interval)
    raise RuntimeError(f"Timed out waiting for Elasticsearch: {last_error}")


def should_force_rebuild():
    return os.getenv("FORCE_REBUILD_INDEX", "").strip().lower() in {"1", "true", "yes"}


@contextmanager
def index_build_lock(lock_path=INDEX_BUILD_LOCK_PATH):
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        print(f"Waiting for index build lock: {lock_path}")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def get_existing_index_state(index_name, timeout=60, interval=2):
    deadline = time.time() + timeout
    last_error = None

    while time.time() < deadline:
        try:
            if not ES_CLIENT.indices.exists(index=index_name):
                return None, None
            ES_CLIENT.cluster.health(
                index=index_name,
                wait_for_status="yellow",
                timeout=f"{max(1, int(interval))}s",
            )
            count = ES_CLIENT.count(index=index_name)["count"]
            meta = get_current_index_meta(ES_CLIENT, index_name)
            return count, meta
        except Exception as exc:
            last_error = exc
            print(f"Waiting for index '{index_name}' to become queryable: {last_error}")
            time.sleep(interval)

    raise RuntimeError(
        f"Timed out waiting for index '{index_name}' to become queryable: {last_error}"
    )


def create_index_and_bulk_index():
    """Create or rebuild the documents index when the fingerprint changes."""
    wait_for_elasticsearch()
    with index_build_lock():
        dataset, mappings, expected_meta = get_expected_index_spec()
        force_rebuild = should_force_rebuild()

        if ES_CLIENT.indices.exists(index=INDEX_NAME):
            count, current_meta = get_existing_index_state(INDEX_NAME)
            if count > 0 and not force_rebuild and index_meta_matches(current_meta, expected_meta):
                print(
                    f"✓ Index '{INDEX_NAME}' already matches expected fingerprint "
                    f"{expected_meta['index_fingerprint'][:12]} with {count} documents. Skipping rebuild."
                )
                return

            rebuild_reason = "force rebuild requested"
            if not force_rebuild:
                if count <= 0:
                    rebuild_reason = "existing index is empty"
                else:
                    rebuild_reason = (
                        "fingerprint mismatch "
                        f"(current={current_meta.get('index_fingerprint')}, "
                        f"expected={expected_meta['index_fingerprint']})"
                    )
            print(f"Rebuilding index '{INDEX_NAME}': {rebuild_reason}")
            ES_CLIENT.indices.delete(index=INDEX_NAME)

        print(
            f"Creating index '{INDEX_NAME}' from dataset {dataset['path']} "
            f"(processed={dataset['processed']})..."
        )
        ES_CLIENT.indices.create(index=INDEX_NAME, mappings=mappings)
        print(f"Index '{INDEX_NAME}' created.")

        documents = load_documents(dataset["path"])
        actions = [
            {
                "_index": INDEX_NAME,
                "_id": doc.get("id", ""),
                "_source": build_source_document(doc, dataset["processed"]),
            }
            for doc in documents
        ]

        bulk(ES_CLIENT, actions, refresh=True)
        print(
            f"Bulk indexing completed. Indexed {len(actions)} documents with fingerprint "
            f"{expected_meta['index_fingerprint'][:12]}."
        )


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
