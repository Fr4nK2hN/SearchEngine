import os
from dataclasses import dataclass


ALLOWED_ROUTER_MODES = frozenset({"baseline", "ltr", "cross_encoder", "hybrid"})
DEFAULT_RECALL_RELAX_THRESHOLD = 5
DEFAULT_ADAPTIVE_HARD_THRESHOLD = 0.6062
DEFAULT_ADAPTIVE_HARD_TOP_K_CAP = 5


def _parse_int(value, default, minimum=None):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def _parse_float(value, default, minimum=None, maximum=None):
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None and parsed < minimum:
        return default
    if maximum is not None and parsed > maximum:
        return default
    return parsed


def _normalize_router_mode(value, default):
    normalized = (value or default or "").strip().lower()
    if normalized not in ALLOWED_ROUTER_MODES:
        return default
    return normalized


@dataclass(frozen=True)
class SearchEngineConfig:
    es_host: str
    es_port: int
    es_scheme: str
    index_name: str
    log_dir: str
    log_filename: str
    cross_encoder_model_name: str
    bi_encoder_model_name: str
    ltr_model_path: str
    query_router_model_path: str
    adaptive_easy_mode: str
    adaptive_hard_mode: str
    adaptive_hard_threshold: float
    adaptive_hard_top_k_cap: int
    recall_relax_threshold: int

    @property
    def elasticsearch_hosts(self):
        return [{"host": self.es_host, "port": self.es_port, "scheme": self.es_scheme}]

    @property
    def log_file(self):
        return os.path.join(self.log_dir, self.log_filename)


def load_config(
    *,
    es_host_default="elasticsearch",
    es_port_default=9200,
    es_scheme_default="http",
):
    es_host = (
        os.getenv("SEARCHENGINE_ES_HOST")
        or os.getenv("ELASTICSEARCH_HOST")
        or es_host_default
    )
    es_port = _parse_int(
        os.getenv("SEARCHENGINE_ES_PORT") or os.getenv("ELASTICSEARCH_PORT"),
        es_port_default,
        minimum=1,
    )
    es_scheme = (os.getenv("SEARCHENGINE_ES_SCHEME") or es_scheme_default).strip() or es_scheme_default

    adaptive_easy_mode = _normalize_router_mode(
        os.getenv("ADAPTIVE_EASY_MODE"),
        "baseline",
    )
    adaptive_hard_mode = _normalize_router_mode(
        os.getenv("ADAPTIVE_HARD_MODE"),
        "cross_encoder",
    )

    return SearchEngineConfig(
        es_host=es_host,
        es_port=es_port,
        es_scheme=es_scheme,
        index_name="documents",
        log_dir=os.getenv("SEARCHENGINE_LOG_DIR", "logs"),
        log_filename=os.getenv("SEARCHENGINE_LOG_FILENAME", "events.log"),
        cross_encoder_model_name=os.getenv(
            "SEARCHENGINE_CROSS_ENCODER_MODEL",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
        ),
        bi_encoder_model_name=os.getenv(
            "SEARCHENGINE_BI_ENCODER_MODEL",
            "all-MiniLM-L6-v2",
        ),
        ltr_model_path=os.getenv("SEARCHENGINE_LTR_MODEL_PATH", "models/ltr_model.pkl"),
        query_router_model_path=os.getenv(
            "SEARCHENGINE_QUERY_ROUTER_MODEL_PATH",
            "models/query_router.pkl",
        ),
        adaptive_easy_mode=adaptive_easy_mode,
        adaptive_hard_mode=adaptive_hard_mode,
        adaptive_hard_threshold=_parse_float(
            os.getenv("ADAPTIVE_HARD_THRESHOLD"),
            DEFAULT_ADAPTIVE_HARD_THRESHOLD,
            minimum=0.0,
            maximum=1.0,
        ),
        adaptive_hard_top_k_cap=_parse_int(
            os.getenv("ADAPTIVE_HARD_TOP_K_CAP"),
            DEFAULT_ADAPTIVE_HARD_TOP_K_CAP,
            minimum=1,
        ),
        recall_relax_threshold=_parse_int(
            os.getenv("RECALL_RELAX_THRESHOLD"),
            DEFAULT_RECALL_RELAX_THRESHOLD,
            minimum=1,
        ),
    )
